//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToRVTensor/RVTLegalizeCommon.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

#include <climits>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>

#include "mlir/Dialect/Quant/QuantTypes.h" // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h" // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/IR/Matchers.h"              // from @llvm-project
#include "mlir/IR/PatternMatch.h"          // from @llvm-project
#include "llvm/Support/FormatVariadic.h"

namespace mlir {
namespace rvtensor {

using namespace mlir::torch::Torch;

std::optional<Value>
createOneDimTfIndices(PatternRewriter &rewriter, Operation *op,
                      SmallVector<int64_t> indicesOneDimShape, int32_t dim,
                      ArrayRef<int64_t> indexShape) {
  unsigned indexRank = indexShape.size();
  SmallVector<int32_t> indicesVec;         // input vec to create rvtensorConstant
  SmallVector<int32_t> indicesMetaElement; // torch.meshgrid inputs
  int indicesMetaElementRepeatTimes{1};    // For torch.stack(torch.meshgrid)

  // Create torch.meshgrid inputs
  // Example: indexShape=[1,4,2]
  // dim0: indicesMetaElement = torch.arange(0, 1) = [0]
  // dim1: indicesMetaElement = torch.arange(0, 4) = [0,1,2,3]
  // dim2: indicesMetaElement = torch.arange(0, 2) = [0,1]
  for (int i = 0; i < indexShape[dim]; i++) {
    indicesMetaElement.push_back(i);
  }

  // Compute total number of meta element repeat times:
  // = product(indexShape[0:dim]) x product(indexShape[dim+1:-1]), skip dim
  // dim0: indicesMetaElementRepeatTimes = 1      x 4*2 = 8
  // dim1: indicesMetaElementRepeatTimes = 1 *1   x   2 = 2
  // dim2: indicesMetaElementRepeatTimes = 1 *1*4       = 4
  for (int i = 0; i < static_cast<int>(indexRank); i++) {
    if (i == dim) {
      continue;
    } else {
      indicesMetaElementRepeatTimes *= indexShape[i];
    }
  }

  if (dim != static_cast<int>(indexShape.size()) - 1) {
    // Create one dim indices for index except for last dim
    // Create indices raw vector.
    // torch.stack(torch.meshgrid)
    // dim0: indicesVec = [0 0 0 0 0 0 0 0]
    // dim0: indicesVec = [0 0 1 1 2 2 3 3]
    for (size_t elementId = 0; elementId < indicesMetaElement.size();
         elementId++) {
      for (int i = 0; i < indicesMetaElementRepeatTimes; i++) {
        indicesVec.push_back(indicesMetaElement[elementId]);
      }
    }
  } else { // Create the one dim indices for last dim of index
    // Create indices raw vector
    // dim2: indicesVec= [0 1 0 1 0 1 0 1]
    // Caution: indicesVec != [0 0 0 0 1 1 1 1]
    for (int i = 0; i < indicesMetaElementRepeatTimes; i++) {
      for (size_t elementId = 0; elementId < indicesMetaElement.size();
           elementId++) {
        indicesVec.push_back(indicesMetaElement[elementId]);
      }
    }
  }

  // Create rvtensor::ConstOp Tensor for indicesVec with target shape.
  // torch.unsqueeze(torch.stack(torch.meshgrid)))
  // dim0:          tensor([[   [ [0], [0] ],
  //			            	[ [0], [0] ],
  //			            	[ [0], [0] ],
  //			              	[ [0], [0] ], ]]) 1*4*2*1
  // dim1:	        tensor([[   [ [0], [0] ],
  //			             	[ [1], [1] ],
  //			            	[ [2], [2] ],
  //			             	[ [3], [3] ], ]]) 1*4*2*1
  // dim2/last dim:	tensor([[   [ [0], [1] ],
  //		                   	[ [0], [1] ],
  //			            	[ [0], [1] ],
  //		    	        	[ [0], [1] ], ]]) 1*4*2*1
  auto indicesDim = getConstTensor<int32_t>(rewriter, op,
                                            /*vec=*/indicesVec,
                                            /*shape=*/indicesOneDimShape);
  return indicesDim;
}

std::optional<Value> convertTorchIndexToTfIndices(PatternRewriter &rewriter,
                                                  Operation *op,
                                                  Value paramsValue,
                                                  Value indexValue,
                                                  int32_t axis) {
  // For easy understanding of this algorithm, the following comments are with
  // an exact example: torch.aten.gather(!torch.vtensor<[1,4,3],f32>, axis=2,
  // !torch.vtensor<[1,4,2],si64>) -> !torch.vtensor<[1,4,2],f32>
  // https://gist.github.com/AmosLewis/2f18434397025211da4491735bcc6db6
  //
  // Convert Torch Index     to       TF Indices
  //    [[                         [[   d0 d1  d2  d0 d1 d2
  //        [0,0],                     [[0, 0, 0],[0, 0, 0]],
  //        [1,0],                     [[0, 1, 1],[0, 1, 0]],
  //        [2,1],                     [[0, 2, 2],[0, 2, 1]],
  //        [2,1]                      [[0, 3, 2],[0, 3, 1]]
  //    ]] 1*4*2                   ]] 1*4*2*3

  auto paramsType = paramsValue.getType().dyn_cast<RankedTensorType>();
  auto indexType = indexValue.getType().dyn_cast<RankedTensorType>();
  auto paramsShape = paramsType.getShape(); // [1 4 3]
  auto indexShape = indexType.getShape();   // [1 4 2]
  int paramsRank = paramsShape.size();      // 3
  int indexRank = indexShape.size();        // 3

  // Initialize the final tf indices shape, and the shape of each dim that can
  // concat to this tf indices
  SmallVector<int64_t> indicesShape;       // [1 4 2 3]
  SmallVector<int64_t> indicesOneDimShape; // [1 4 2 1]
  for (auto shape : indexShape) {
    indicesShape.push_back(shape);
    indicesOneDimShape.push_back(shape);
  }
  indicesShape.push_back(paramsRank);
  indicesOneDimShape.push_back(1);

  // Get the chosen axis index
  // indexValue reshape to indicesDim: shape append 1
  // [1 4 2] -> [1 4 2 1]
  // dim2:	tensor([[   [ [0], [0] ],
  //			    [ [1], [0] ],
  //			    [ [2], [1] ],
  //			    [ [2], [1] ], ]]) 1*4*2*1
  auto indicesChosenAxis = rvtensor::CreateOpAndInfer<rvtensor::ReshapeOp>(
      rewriter, op->getLoc(),
      GetTypeFromTensorShape(indicesOneDimShape, indexType.getElementType()),
      indexValue, rewriter.getDenseI64ArrayAttr(indicesOneDimShape));

  SmallVector<Value> concatInputs;
  for (auto dim = 0; dim < paramsRank; dim++) {
    if (dim != axis) {
      auto indices = createOneDimTfIndices(rewriter, op, indicesOneDimShape,
                                           dim, indexShape);
      concatInputs.push_back(indices.value());
    } else {
      // the chosen axis indices will be replaced by index[i][j][k]
      concatInputs.push_back(indicesChosenAxis.getResult());
    }
  }

  // detailed example explanation
  // https://gist.github.com/AmosLewis/932a8dee3ba7657dcc6d09a4da4775d4 Get TF
  // indices: 1*4*2*3
  // [[  d0 d1  d2  d0 d1 d2
  //    [[0, 0, 0],[0, 0, 0]],
  //    [[0, 1, 1],[0, 1, 0]],
  //    [[0, 2, 2],[0, 2, 1]],
  //    [[0, 3, 2],[0, 3, 1]]
  // ]]
  auto indicesTf = rvtensor::CreateOpAndInfer<rvtensor::ConcatOp>(
      rewriter, op->getLoc(),
      GetTypeFromTensorShape(indicesShape, rewriter.getIntegerType(32)),
      concatInputs, indexRank);

  return indicesTf.getResult();
}

} // namespace rvtensor
} // namespace mlir
