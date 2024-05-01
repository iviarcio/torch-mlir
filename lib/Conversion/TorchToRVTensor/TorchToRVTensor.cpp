//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToRVTensor/TorchToRVTensor.h"
#include "../PassDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "rvtensor/RVTensor/RVTensorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Conversion/TorchToRVTensor/RVTLegalizeCommon.h"
#include "torch-mlir/Conversion/TorchToRVTensor/RVTLegalizeUtils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"
#include <optional>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

template <typename T>
static bool isInValidRange(bool isFloat, const double &doubleValue, bool isInt,
                           const int64_t &intValue) {
  if (isFloat) {
    // Do a round-trip check here instead of numeric limits due to
    // compiler warnings around double <-> int conversion.
    return (doubleValue == static_cast<double>(static_cast<T>(doubleValue)));
  } else {
    assert(isInt);
    return (intValue >= static_cast<int64_t>(std::numeric_limits<T>::min())) &&
           (intValue <= static_cast<int64_t>(std::numeric_limits<T>::max()));
  }
  return true;
}

// FIXME: This will eventually go into a RVTensor*Utils file.
LogicalResult torchScalarToRVTensorTensor(ConversionPatternRewriter &rewriter,
                                      Operation *op, Value torchScalarValue,
                                      Value &rvtensorTensor, Type dtype,
                                      llvm::ArrayRef<int64_t> dshape) {
  // Retrieve a const float or int value but create the out Tensor with dtype.
  double doubleValue;
  auto isFloat =
      matchPattern(torchScalarValue, m_TorchConstantFloat(&doubleValue));

  int64_t intValue;
  auto isInt = matchPattern(torchScalarValue, m_TorchConstantInt(&intValue));

  if (!isFloat && !isInt)
    return rewriter.notifyMatchFailure(op,
                                       "Unable to extract the scalar constant");

  if (isa<mlir::FloatType>(dtype)) {
    rvtensorTensor = rvtensor::getConstTensor<float>(rewriter, op,
                                             (isFloat ? doubleValue : intValue),
                                             dshape, dtype)
                     .value();
  } else if (auto intType = dyn_cast<mlir::IntegerType>(dtype)) {
    auto w = intType.getWidth();
    if (w != 1 && w != 32 && w != 64)
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag << "Unsupported integer type: " << intType;
      });

    if (w == 1) {
      if (!isInValidRange<bool>(isFloat, doubleValue, isInt, intValue)) {
        return rewriter.notifyMatchFailure(
            op, "Supplied value of scalar constant exceeds limits "
                "of destination type");
      }
      bool d = isFloat ? static_cast<bool>(doubleValue)
                       : static_cast<bool>(intValue);
      rvtensorTensor =
          rvtensor::getConstTensor<bool>(rewriter, op, {d}, dshape).value();
    } else if (w == 32) {
      if (!isInValidRange<int32_t>(isFloat, doubleValue, isInt, intValue)) {
        return rewriter.notifyMatchFailure(
            op, "Supplied value of scalar constant exceeds limits "
                "of destination type");
      }
      int32_t d = isFloat ? static_cast<int32_t>(doubleValue)
                          : static_cast<int32_t>(intValue);
      rvtensorTensor =
          rvtensor::getConstTensor<int32_t>(rewriter, op, {d}, dshape).value();
    } else if (w == 64) {
      if (!isInValidRange<int64_t>(isFloat, doubleValue, isInt, intValue)) {
        return rewriter.notifyMatchFailure(
            op, "Supplied value of scalar constant exceeds limits "
                "of destination type");
      }
      int64_t d = (isFloat ? static_cast<int64_t>(doubleValue) : intValue);
      rvtensorTensor =
          rvtensor::getConstTensor<int64_t>(rewriter, op, {d}, dshape).value();
    }
  } else {
    return rewriter.notifyMatchFailure(op, "Usupported element type");
  }

  return success();
}

LogicalResult torchAlphaToRVTensorTensor(ConversionPatternRewriter &rewriter,
                                     Operation *op, Value alphaScalar,
                                     Value &alphaTensor, Type dtype,
                                     bool checkForUnity) {
  if (succeeded(torchScalarToRVTensorTensor(rewriter, op, alphaScalar, alphaTensor,
                                        dtype, {})))
    return success();

  // `alpha` has not been specified.
  int64_t alphaValue;
  if (!matchPattern(alphaScalar, m_TorchConstantInt(&alphaValue)))
    return rewriter.notifyMatchFailure(
        op, "Currently only scalar constants are supported for "
            "alpha in RVTENSOR operation");
  // When no alpha has been specified, this must be 1.
  if (checkForUnity && alphaValue != 1)
    return rewriter.notifyMatchFailure(op,
                                       "Unsupported integer value for alpha");

  alphaTensor = rvtensor::getConstTensor<float>(
                    rewriter, op, {static_cast<float>(alphaValue)}, {}, dtype)
                    .value();

  return success();
}

// Binary op legalizations for comparator ops.
template <typename AtenOpT, typename RVTensorOpT>
class ConvertAtenCompareOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getSelf();
    auto lhsTy = lhs.getType().dyn_cast<TensorType>();
    Value rhs = adaptor.getOther();
    auto rhsTy = rhs.getType().dyn_cast<TensorType>();

    if (!lhsTy)
      return rewriter.notifyMatchFailure(op,
                                         "Only Tensor types supported in RVTENSOR");

    auto lhsElemTy = lhsTy.getElementType();
    if (!lhsElemTy.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "Only floating-point or integer datatype legalization supported");

    // For bitwise operators, only integer datatype legalization is supported
    constexpr bool isBitwiseOp =
        std::is_same<AtenOpT, AtenBitwiseAndTensorOp>() ||
        std::is_same<AtenOpT, AtenBitwiseOrTensorOp>() ||
        std::is_same<AtenOpT, AtenBitwiseXorTensorOp>();
    if (isa<mlir::FloatType>(lhsElemTy) && isBitwiseOp) {
      return rewriter.notifyMatchFailure(op,
                                         "For bitwise operators, only integer "
                                         "datatype legalization is supported");
    }

    Value rhsAsTensor;
    if (!rhsTy) {
      if (failed(torchScalarToRVTensorTensor(rewriter, op, op.getOther(),
                                         rhsAsTensor, lhsElemTy, {})))
        return rewriter.notifyMatchFailure(
            op, "Currently only scalar constants are supported for "
                "conversion in RVTENSOR operation");
    }
    auto rhsTensor = rhsTy ? rhs : rhsAsTensor;
    // There is no Lesser operator in RVTENSOR.
    auto swapLhsRhs = (std::is_same<AtenOpT, AtenLtTensorOp>() ||
                       std::is_same<AtenOpT, AtenLtScalarOp>());

    // Promote lhs and rhs dtypes for bitwise operators.
    TensorType resultTy = OpConversionPattern<AtenOpT>::getTypeConverter()
                              ->convertType(op.getType())
                              .template cast<TensorType>();
    if (isBitwiseOp) {
      lhs = rvtensor::promoteType(rewriter, lhs, resultTy);
      rhsTensor = rvtensor::promoteType(rewriter, rhsTensor, resultTy);
    }

    auto resultOp = rewriter.create<RVTensorOpT>(op.getLoc(), resultTy,
                                             (swapLhsRhs ? rhsTensor : lhs),
                                             (swapLhsRhs ? lhs : rhsTensor));

    // There is no NE operator in RVTENSOR.
    if (std::is_same<AtenOpT, AtenNeTensorOp>() ||
        std::is_same<AtenOpT, AtenNeScalarOp>())
      rewriter.replaceOpWithNewOp<rvtensor::LogicalNotOp>(op, resultTy,
                                                      resultOp.getResult());
    else
      rewriter.replaceOp(op, resultOp.getResult());

    return success();
  }
};

// This defines a template to construct ops whose legalizations are
// specialized.
template <typename AtenOpT>
class ConvertAtenOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

using ReductionConvFunc = std::optional<Value> (*)(PatternRewriter &,
                                                   Operation *,
                                                   RankedTensorType, Value,
                                                   ElementsAttr, bool);

// They all constitute a common form invoking the appropriate
// conversion function in RVTLegalizeCommon.cpp
template <typename AtenOpT, ReductionConvFunc ConversionFuncT>
class ConvertAtenReductionOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;

  // Each variant must implement corresponding parameter parsing options
  virtual LogicalResult readReduceDimsAndKeepDims(
      AtenOpT op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter,
      ElementsAttr &reduceDimsAttr, bool &keepDims) const {
    return rewriter.notifyMatchFailure(
        op, "Unimplemented reduce_dims and keep_dims parsing function");
  }

  // Common rewriter for all reduction ops, calls the specific implementation of
  // readReduceDimsAndKeepDims() needed for the op variant.
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.getSelf();
    auto selfTy = self.getType().cast<TensorType>();

    if (!selfTy)
      return rewriter.notifyMatchFailure(op,
                                         "Only Tensor types supported in RVTENSOR");

    auto outputTy = OpConversionPattern<AtenOpT>::getTypeConverter()
                        ->convertType(op.getType())
                        .template cast<RankedTensorType>();
    if (!outputTy)
      return rewriter.notifyMatchFailure(
          op, "Only ranked tensor type outputs permitted for reduce_mean");

    ElementsAttr reduceDimsAttr;
    bool keepDims;

    if (failed(readReduceDimsAndKeepDims(op, adaptor, rewriter, reduceDimsAttr,
                                         keepDims)))
      return failure();

    std::optional<Value> result =
        ConversionFuncT(rewriter, op, outputTy, self, reduceDimsAttr, keepDims);

    if (!result)
      return failure();

    // TBD - support dtype casting.

    rewriter.replaceOp(op, {result.value()});

    return success();
  }
};

// This reduction op legalization template handles op variants that have
// explicit reduce_dims dimensions (provided as a list) and keep_dims
// parameters.
template <typename AtenOpT, ReductionConvFunc ConversionFuncT>
class ConvertAtenMultipleDimsReductionOp
    : public ConvertAtenReductionOp<AtenOpT, ConversionFuncT> {
  using ConvertAtenReductionOp<AtenOpT,
                               ConversionFuncT>::ConvertAtenReductionOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult readReduceDimsAndKeepDims(AtenOpT op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter,
                                          ElementsAttr &reduceDimsAttr,
                                          bool &keepDims) const override {
    SmallVector<int64_t, 4> reduceDims;
    if (!matchPattern(op.getDim(), m_TorchListOfConstantInts(reduceDims)))
      return rewriter.notifyMatchFailure(op,
                                         "non-const dim parameter unsupported");
    int64_t N = reduceDims.size();
    int64_t inputRank =
        adaptor.getSelf().getType().template cast<RankedTensorType>().getRank();
    for (unsigned i = 0; i < N; i++) {
      reduceDims[i] = toPositiveDim(reduceDims[i], inputRank);
      if (!isValidDim(reduceDims[i], inputRank))
        return rewriter.notifyMatchFailure(op,
                                           "reduce dim is statically invalid");
    }
    auto reduceDimsType = RankedTensorType::get({N}, rewriter.getI64Type());
    reduceDimsAttr =
        DenseIntElementsAttr::get(reduceDimsType, llvm::ArrayRef(reduceDims));

    keepDims = false;
    if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepDims)))
      return rewriter.notifyMatchFailure(
          op, "non-const keepdim parameter unsupported");

    return success();
  }
};

// This reduction op legalization template handles op variants that reduce in
// only one explicit dim which is provided as a number (rather than a list), and
// a keep_dims parameter.
template <typename AtenOpT, ReductionConvFunc ConversionFuncT>
class ConvertAtenOneDimReductionOp
    : public ConvertAtenReductionOp<AtenOpT, ConversionFuncT> {
  using ConvertAtenReductionOp<AtenOpT,
                               ConversionFuncT>::ConvertAtenReductionOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult readReduceDimsAndKeepDims(AtenOpT op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter,
                                          ElementsAttr &reduceDimsAttr,
                                          bool &keepDims) const override {
    int64_t reduceDim;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&reduceDim)))
      return rewriter.notifyMatchFailure(op,
                                         "non-const dim parameter unsupported");
    int64_t inputRank =
        adaptor.getSelf().getType().template cast<RankedTensorType>().getRank();
    reduceDim = toPositiveDim(reduceDim, inputRank);
    if (!isValidDim(reduceDim, inputRank))
      return rewriter.notifyMatchFailure(op, "dim is statically invalid");
    auto reduceDimsType = RankedTensorType::get({1}, rewriter.getI64Type());
    reduceDimsAttr =
        DenseIntElementsAttr::get(reduceDimsType, llvm::ArrayRef({reduceDim}));

    keepDims = false;
    if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepDims)))
      return rewriter.notifyMatchFailure(
          op, "non-const keepdim parameter unsupported");

    return success();
  }
};

// This reduction op legalization template handles op variants that reduce all
// dims does not keep dims.
template <typename AtenOpT, ReductionConvFunc ConversionFuncT>
class ConvertAtenAllDimsReductionOp
    : public ConvertAtenReductionOp<AtenOpT, ConversionFuncT> {
public:
  using ConvertAtenReductionOp<AtenOpT,
                               ConversionFuncT>::ConvertAtenReductionOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult readReduceDimsAndKeepDims(AtenOpT op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter,
                                          ElementsAttr &reduceDimsAttr,
                                          bool &keepDims) const override {
    auto self = adaptor.getSelf();
    auto selfTy = self.getType().template cast<RankedTensorType>();

    // Select all dims to reduce
    SmallVector<int64_t, 4> reduceDims;
    for (int64_t i = 0; i < selfTy.getRank(); i++)
      reduceDims.push_back(i);
    int64_t N = selfTy.getRank();
    auto reduceDimsType = RankedTensorType::get({N}, rewriter.getI64Type());
    reduceDimsAttr =
        DenseIntElementsAttr::get(reduceDimsType, llvm::ArrayRef(reduceDims));
    keepDims = false;

    return success();
  }
};

// Perform the basic n-dim matmul operation encompassing the handling of
// broadcasting and dynamic shape propagation.
// All PyTorch ops that leverage matrix multiplication will derive this and
// implement their specialized input processing (e.g transpose), and output
// processing, e.g. GEMM or fully connected bias handling.
template <typename AtenOpT>
class ConvertAtenMatmulBaseOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  // Each variant must implement corresponding parameter parsing options.
  // Maintain separate input read functions for each variant because it is not
  // necessarily true with all variants that the first two operands are the lhs
  // and rhs.
  virtual LogicalResult readMatMulInputs(AtenOpT op, OpAdaptor adaptor,
                                         ConversionPatternRewriter &rewriter,
                                         Value &lhs, Value &rhs) const {
    return rewriter.notifyMatchFailure(
        op,
        "Unimplemented matrix multiplication variant input parsing function");
  }
  LogicalResult performMatmul(AtenOpT op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter, Value &lhs,
                              Value &rhs, Value &output) const {

    auto lhsTy = lhs.getType().cast<RankedTensorType>();
    auto rhsTy = rhs.getType().cast<RankedTensorType>();

    auto lhsRank = lhsTy.getRank();
    auto rhsRank = rhsTy.getRank();

    auto lhsShape = makeShapeTorchCompatible(lhsTy.getShape());
    auto rhsShape = makeShapeTorchCompatible(rhsTy.getShape());

    auto lhsElemTy = lhsTy.getElementType();
    auto rhsElemTy = rhsTy.getElementType();

    if (lhsElemTy != rhsElemTy)
      return rewriter.notifyMatchFailure(op,
                                         "Matmul: input datatypes mismatched");

    // Legalization constructs may offer input shapes but expect output shapes
    // to be inferred, e.g.
    // func @forward(%arg0: !torch.vtensor<[14,19],f32>,
    //               %arg1: !torch.vtensor<[19,28],f32>) ->
    //               !torch.vtensor<[?,?],f32>
    // This is tricky with matmul, since RVTENSOR matmul is on 3D inputs.
    // This means the need to reshape potentially both inputs and outputs,
    // and reshape to unknown shape is undefined.

    auto maxInputRank = lhsRank > rhsRank ? lhsRank : rhsRank;
    // If performing dot product on vectors, the RHS is synthetically transposed
    if (maxInputRank == 1)
      maxInputRank++;

    // Obtaining the rank broadcasted shapes of tensors makes it easier to
    // construct the input and output reshaping logic.
    auto getRankBroadcastedShape = [&](Value tensor,
                                       bool isRHS) -> SmallVector<int64_t> {
      auto tensorTy = tensor.getType().cast<TensorType>();
      auto tensorShape = makeShapeTorchCompatible(tensorTy.getShape());
      auto tensorRank = tensorTy.getRank();

      SmallVector<int64_t> bcastedShape;

      auto bcastDims = maxInputRank - tensorRank;

      if (isRHS && (tensorRank == 1) && bcastDims) {
        // RHS with rank1 is special. It be synthetically transposed to dim[:-2]
        for (int32_t i = 0; i < bcastDims - 1; i++)
          bcastedShape.push_back(1);
        bcastedShape.push_back(tensorShape[0]);
        bcastedShape.push_back(1);
      } else {
        if (bcastDims > 0) { // rank broadcast
          for (uint32_t i = 0; i < bcastDims; i++)
            bcastedShape.push_back(1);
        }
        for (auto &dim : tensorShape)
          bcastedShape.push_back(dim);
      }
      return bcastedShape;
    };

    // Step: Rank broadcast the two inputs.
    auto lhsBroadcastedShape = getRankBroadcastedShape(lhs, false);
    auto lhsBroadcastedTy = RankedTensorType::get(
        makeShapeLLVMCompatible(lhsBroadcastedShape), lhsElemTy);
    auto rhsBroadcastedShape = getRankBroadcastedShape(rhs, true);
    auto rhsBroadcastedTy = RankedTensorType::get(
        makeShapeLLVMCompatible(rhsBroadcastedShape), rhsElemTy);

    auto rankBroadcastedLhs =
        lhsRank == maxInputRank
            ? lhs
            : rewriter.create<rvtensor::ReshapeOp>(
                  op->getLoc(),
                  OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
                      lhsBroadcastedTy),
                  lhs, rewriter.getDenseI64ArrayAttr(lhsBroadcastedShape));

    auto rankBroadcastedRhs =
        rhsRank == maxInputRank
            ? rhs
            : rewriter.create<rvtensor::ReshapeOp>(
                  op->getLoc(),
                  OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
                      rhsBroadcastedTy),
                  rhs, rewriter.getDenseI64ArrayAttr(rhsBroadcastedShape));

    // RVTENSOR matmul is performed on two 3D inputs and generates a 3D output.
    // Lower ranked tensors are dim-1 reshaped up to 3D
    auto reshapeUpTo3DTensor = [&](Value tensor) -> Value {
      auto tensorTy = tensor.getType().cast<TensorType>();
      auto rank = tensorTy.getRank();

      assert(rank <= 3 && "reshapeUpTo3D tensor must receive rank <= 3");
      if (rank == 3)
        return tensor;

      auto shape = makeShapeTorchCompatible(tensorTy.getShape());
      SmallVector<int64_t> newShape({1, 1, 1});

      if (rank == 2) { // batchsize = 1
        newShape[1] = shape[0];
        newShape[2] = shape[1];
      } else { // rank 1
        newShape[2] = shape[0];
      }
      auto newType = RankedTensorType::get(makeShapeLLVMCompatible(newShape),
                                           tensorTy.getElementType());

      return rewriter.create<rvtensor::ReshapeOp>(
          op->getLoc(),
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              newType),
          tensor, rewriter.getDenseI64ArrayAttr(newShape));
    };

    // Where broadcasting is required in one or more batch dims, the following
    // is done.
    // Where all batch dims are involved in broadcasting:
    // Given A: 3x1x5x6 and B: 1x4x6x7
    // 1. Reshape A to 1x15x6 (squeeze all batchdims into dim1)
    // 2. Transpose B to 6x1x4x7, Reshape to 1x6x28
    // 3. rvtensor.Matmul 1x15x6 1x6x28 = 1x15x28
    // 4. Reshape out to 3x5x4x7, Transpose to 3x4x5x7
    // Where there are batch dimensions that are broadcast and not, the
    // treatment is to have dim0 correspond to product of all non-broadcast
    // dimsizes:
    // Given A: 4x8x16x32 B: 8x32x17
    // 1. Reshape A to 8x64x32 (squeeze all unbroadcasted dims into dim0,
    // broadcasted dims into dim1)
    // 2. No transpose or reshape of B as its batchdims are not broadcast to.
    // 3. rvtensor.Matmul 8x64x32 8x32x17 = 8x64x17
    // 4. Reshape to 8x4x16x17, Transpose to 4x8x16x17

    // Check if we need to perform the broadcast on batch dim
    // Not needed if max rank < 3, or if maxrank == 3 and dim[0] matches
    auto needsBatchDimBroadcast = [&]() -> bool {
      if (maxInputRank < 3) {
        return false;
      } else {
        if (maxInputRank == 3 &&
            lhsBroadcastedShape[0] == rhsBroadcastedShape[0]) {
          return false;
        }
        return true;
      }
    };

    auto performBatchDimBroadcast = needsBatchDimBroadcast();

    // Inputs to the rvtensor.matmul
    Value matmulLhs, matmulRhs;

    using TensorShape_t = struct {
      int64_t dim;
      int64_t shape;
    };

    // Transpose needs to done if transposeDims are not non-monotonically
    // increasing. E.g. [0, 1, 2, 3]: No transpose [1, 0, 2, 3]: Transpose dim0
    // and dim1 The order need not be sequential, since one or more dims may
    // have been removed due to broadcasting.
    auto isTransposeRequired = [](SmallVector<int32_t> transposedDims) -> bool {
      int32_t lastDim = -1;
      for (auto &dim : transposedDims) {
        if (lastDim > dim)
          return true;
        lastDim = dim;
      }
      return false;
    };

    SmallVector<TensorShape_t> batchElems, lhsSqueezedElems, rhsSqueezedElems;

    if (!performBatchDimBroadcast) {
      // Simple with no broadcasting artifacts. Just reshape up to 3D
      matmulLhs = reshapeUpTo3DTensor(rankBroadcastedLhs);
      matmulRhs = reshapeUpTo3DTensor(rankBroadcastedRhs);

    } else {
      // In this case, either or both input matrices involve broadcasting on
      // their batch dimensions. For example:
      // 4x5x6, 1x6x7 -> 4x5x7
      // 4x1x5x6, 1x3x6x7 -> 4x3x5x7
      // Though maxInputRank is necessarily >=3 here, individual matrices may be
      // lower rank.
      // E.g. 3x4x5x6, 6 -> 3x4x5

      // These are the accumulated products of the shape of each dim:
      // 1. common dimensions: upper dimensions (dims other than two rightmost)
      // whose shapes are the same for both LHS and RHS.
      // 2. LHS squeezed dimensions: all dimensions of LHS that involve
      // broadcasting in either direction, plus the LHS[-2] shape
      // 3. RHS squeezed dimensions: all dimensions of RHS that involve
      // broadcasting in either direction, plus the RHS[-1] shape
      int64_t commonValue = 1, lhsSqueezedValue = 1, rhsSqueezedValue = 1;

      // For both LHS and RHS, the dimensions are separated into the common,
      // squeezed and remaining dim. E.g. given
      // LHS = 3x4x5x6
      // RHS = 1x4x6x7
      // common = {{dim=1, shape=4}}
      // lhs squeezed = {{dim=0, shape=3},
      //                 {dim=2, shape=5}}
      // rhs squeezed = {{dim=0, shape=1},
      //                 {dim=2, shape=7}}
      // The matmul dim is LHS[-1] and RHS[-2], i.e. 6.
      // Once this is obtained, LHS and RHS are expressed as:
      // LHS = {common, lhs_squeezed, matmul_dim}
      // RHS = {common, matmul_dim, rhs_squeezed}
      // The matmul is then performed to obtain output:
      // matmul_out = {common, lhs_squeezed, rhs_squeezed}
      // Finally, we reshape to 'unsqueeze' the LHS and RHS parts and transpose
      // them back to their correct positions.

      SmallVector<int64_t> transposedLhsShape;
      SmallVector<int32_t> transposedLhsDims;

      // Step: generate the common dim/shape information
      bool hasDynamicDims = false;
      for (uint32_t dim = 0; dim < maxInputRank - 2; dim++) {
        bool isDynamicDim = ShapedType::isDynamic(lhsBroadcastedShape[dim]);
        hasDynamicDims |= isDynamicDim;
        if (isDynamicDim ||
            lhsBroadcastedShape[dim] == rhsBroadcastedShape[dim]) {
          commonValue *= lhsBroadcastedShape[dim];
          batchElems.push_back({dim, lhsBroadcastedShape[dim]});
        }
      }
      commonValue = commonValue < 0 ? kUnknownSize : commonValue;

      // TODO: Handle the case when there are dynamic batch dimensions.
      if (hasDynamicDims)
        commonValue = kUnknownSize;

      // Step: generate the LHS squeezed dim/shape information.
      for (uint32_t dim = 0; dim < maxInputRank - 2; dim++) {
        bool isDynamicDim = ShapedType::isDynamic(lhsBroadcastedShape[dim]);
        if (!isDynamicDim &&
            lhsBroadcastedShape[dim] != rhsBroadcastedShape[dim]) {
          lhsSqueezedValue *= lhsBroadcastedShape[dim];
          lhsSqueezedElems.push_back({dim, lhsBroadcastedShape[dim]});
        }
      }
      // including LHS[-2]
      lhsSqueezedElems.push_back(
          {maxInputRank - 2, lhsBroadcastedShape[maxInputRank - 2]});
      lhsSqueezedValue *= lhsBroadcastedShape[maxInputRank - 2];
      lhsSqueezedValue = lhsSqueezedValue < 0 ? kUnknownSize : lhsSqueezedValue;

      // Step: Create the rvtensor.transpose array. If this array has a
      // non-monotonic series of dims, perform transpose.
      // First the common_elems
      for (uint32_t i = 0; i < batchElems.size(); i++) {
        transposedLhsShape.push_back(batchElems[i].shape);
        transposedLhsDims.push_back(batchElems[i].dim);
      }
      // then the lhs_squeezed elems
      for (uint32_t i = 0; i < lhsSqueezedElems.size(); i++) {
        transposedLhsShape.push_back(lhsSqueezedElems[i].shape);
        transposedLhsDims.push_back(lhsSqueezedElems[i].dim);
      }
      // then the final dim
      transposedLhsDims.push_back(maxInputRank - 1);
      transposedLhsShape.push_back(lhsBroadcastedShape[maxInputRank - 1]);

      bool lhsNeedsTranspose = isTransposeRequired(transposedLhsDims);

      auto lhsReshapeInput = rankBroadcastedLhs;

      if (lhsNeedsTranspose) {
        auto transposedLhsType = RankedTensorType::get(
            makeShapeLLVMCompatible(transposedLhsShape), rhsElemTy);

        std::optional<Value> transposedLhsDimsConst =
            rvtensor::getConstTensor<int32_t>(
                rewriter, op,
                /*vec=*/transposedLhsDims,
                /*shape=*/{static_cast<int32_t>(transposedLhsDims.size())});

        lhsReshapeInput =
            rewriter
                .create<rvtensor::TransposeOp>(
                    op->getLoc(),
                    OpConversionPattern<AtenOpT>::getTypeConverter()
                        ->convertType(transposedLhsType),
                    rankBroadcastedLhs, transposedLhsDimsConst.value())
                .getResult();
      }

      // LHS = {common, lhs_squeezed, matmul_dim}
      SmallVector<int64_t> newLhsShape(
          {1, 1, lhsBroadcastedShape[maxInputRank - 1]});
      newLhsShape[0] = commonValue;
      newLhsShape[1] = hasDynamicDims ? kUnknownSize : lhsSqueezedValue;

      auto newLhsType = RankedTensorType::get(
          makeShapeLLVMCompatible(newLhsShape), lhsElemTy);

      matmulLhs = rewriter.create<rvtensor::ReshapeOp>(
          op->getLoc(),
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              newLhsType),
          lhsReshapeInput, rewriter.getDenseI64ArrayAttr(newLhsShape));

      SmallVector<int64_t> transposedRhsShape;
      SmallVector<int32_t> transposedRhsDims;

      // Step: Create the RHS transpose sequence
      // RHS = {common, matmul_dim, rhs_squeezed}
      // first the common_dims
      for (uint32_t i = 0; i < batchElems.size(); i++) {
        transposedRhsShape.push_back(batchElems[i].shape);
        transposedRhsDims.push_back(batchElems[i].dim);
      }
      // The matmul_dim of RHS
      transposedRhsDims.push_back(maxInputRank - 2);
      transposedRhsShape.push_back(rhsBroadcastedShape[maxInputRank - 2]);
      // finally all the rhs_squeeze dims
      hasDynamicDims = false;
      for (uint32_t dim = 0; dim < maxInputRank - 2; dim++) {
        bool isDynamicDim = ShapedType::isDynamic(rhsBroadcastedShape[dim]);
        hasDynamicDims |= isDynamicDim;
        if (!isDynamicDim &&
            rhsBroadcastedShape[dim] != lhsBroadcastedShape[dim]) {
          rhsSqueezedElems.push_back({dim, rhsBroadcastedShape[dim]});
          rhsSqueezedValue *= rhsBroadcastedShape[dim];
        }
      }
      rhsSqueezedElems.push_back(
          {maxInputRank - 1, rhsBroadcastedShape[maxInputRank - 1]});
      rhsSqueezedValue *= rhsBroadcastedShape[maxInputRank - 1];
      for (uint32_t i = 0; i < rhsSqueezedElems.size(); i++) {
        transposedRhsShape.push_back(rhsSqueezedElems[i].shape);
        transposedRhsDims.push_back(rhsSqueezedElems[i].dim);
      }

      auto transposedRhsType = RankedTensorType::get(
          makeShapeLLVMCompatible(transposedRhsShape), rhsElemTy);

      if (hasDynamicDims)
        rhsSqueezedValue = kUnknownSize;

      SmallVector<int64_t> newRhsShape(
          {commonValue < 0 ? kUnknownSize : commonValue,
           rhsBroadcastedShape[maxInputRank - 2], rhsSqueezedValue});
      auto newRhsType = RankedTensorType::get(
          makeShapeLLVMCompatible(newRhsShape), rhsElemTy);

      bool rhsNeedsTranspose = isTransposeRequired(transposedRhsDims);

      auto transposedRhsValue = rankBroadcastedRhs;

      if (rhsNeedsTranspose) {
        std::optional<Value> transposedRhsDimsConst =
            rvtensor::getConstTensor<int32_t>(
                rewriter, op,
                /*vec=*/transposedRhsDims,
                /*shape=*/{static_cast<int32_t>(transposedRhsDims.size())});

        transposedRhsValue =
            rewriter
                .create<rvtensor::TransposeOp>(
                    op->getLoc(),
                    OpConversionPattern<AtenOpT>::getTypeConverter()
                        ->convertType(transposedRhsType),
                    rankBroadcastedRhs, transposedRhsDimsConst.value())
                .getResult();
      }

      // reshape
      matmulRhs = rewriter.create<rvtensor::ReshapeOp>(
          op->getLoc(),
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              newRhsType),
          transposedRhsValue, rewriter.getDenseI64ArrayAttr(newRhsShape));
    }

    auto matmulLhsShape = makeShapeTorchCompatible(
        matmulLhs.getType().template cast<RankedTensorType>().getShape());
    auto matmulRhsShape = makeShapeTorchCompatible(
        matmulRhs.getType().template cast<RankedTensorType>().getShape());

    // The reshape/transpose should ensure the rvtensor.matmul always has same
    // batch size for either matrix. If if shapes are dynamic, they'll be
    // appropriately handled.
    assert(matmulLhsShape[0] == matmulRhsShape[0] &&
           "rvtensor.matmul needs same batchsize on LHS and RHS");

    SmallVector<int64_t> matmulOutputShape(
        {matmulLhsShape[0], matmulLhsShape[1], matmulRhsShape[2]});
    Type outputElemTy;
    if (isa<mlir::FloatType>(lhsElemTy)) {
      outputElemTy = lhsElemTy;
    } else { // qint8 emits i32 matmul output
      outputElemTy = rewriter.getIntegerType(32);
    }

    auto mmOutputTy = RankedTensorType::get(
        makeShapeLLVMCompatible(matmulOutputShape), outputElemTy);
    auto mmOpResult =
        rewriter
            .create<rvtensor::MatMulOp>(
                op->getLoc(),
                OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
                    mmOutputTy),
                matmulLhs, matmulRhs)
            .getResult();

    // Perform the reshape to output shape. This is always required unless max
    // input rank=3 and there was no broadcasting, in which case the rvtensor.matmul
    // output itself is correctly shaped.
    bool performOpReshape = !(maxInputRank == 3 && !performBatchDimBroadcast);

    if (performOpReshape) {
      // Since the output shape may be unknown, we construct it
      // independently and reshape. Otherwise reshape may be expressed for
      // an unknown to-be-inferred output shape. The final tensor.cast
      // reshapes the known shape to the desired output shape.
      auto computeOpShape = [&](SmallVector<int64_t> &reshapedOpShape,
                                SmallVector<int32_t> &transposedOpDims,
                                SmallVector<int64_t> &transposedOpShapes) {
        if (maxInputRank == 1)
          return;

        if (maxInputRank == 2) {
          if (lhsRank == 2)
            reshapedOpShape.push_back(lhsShape[0]);
          if (rhsRank == 2)
            reshapedOpShape.push_back(rhsShape[1]);
          return;
        }

        // Step: Construct the output transpose/reshape information
        // First the common_dims
        for (uint32_t i = 0; i < batchElems.size(); i++) {
          reshapedOpShape.push_back(batchElems[i].shape);
          transposedOpDims.push_back(batchElems[i].dim);
        }

        // Then the LHS squeezed dims
        for (uint32_t i = 0; i < lhsSqueezedElems.size() - 1; i++) {
          // Only dims that don't broadcast - broadcasting ones come from the
          // other input.
          if (lhsSqueezedElems[i].shape != 1) {
            reshapedOpShape.push_back(lhsSqueezedElems[i].shape);
            transposedOpDims.push_back(lhsSqueezedElems[i].dim);
          }
        }
        // The last squeezed dim is lhs[-2] which needs to be
        // checked separately for broadcasting
        if (lhsRank > 1) {
          reshapedOpShape.push_back(lhsBroadcastedShape[maxInputRank - 2]);
          transposedOpDims.push_back(maxInputRank - 2);
        }

        // then the RHS squeezed dims except rhs[-1] which is handled like
        // lhs[-2]
        for (uint32_t i = 0; i < rhsSqueezedElems.size() - 1; i++) {
          if (rhsSqueezedElems[i].shape != 1) {
            reshapedOpShape.push_back(rhsSqueezedElems[i].shape);
            transposedOpDims.push_back(rhsSqueezedElems[i].dim);
          }
        }
        // rhs[-1]
        if (rhsRank > 1) {
          reshapedOpShape.push_back(rhsBroadcastedShape[maxInputRank - 1]);
          transposedOpDims.push_back(maxInputRank - 1);
        }

        // The transposition order is the inverse of what we actually want,
        // inversing should fix this:
        llvm::SmallVector<int> inverseTransposeDims(transposedOpDims.size());
        for (int i = 0, s = transposedOpDims.size(); i < s; ++i)
          inverseTransposeDims[transposedOpDims[i]] = i;

        transposedOpDims = inverseTransposeDims;

        // Final transposed output shape construction
        for (uint32_t i = 0; i < maxInputRank - 2; i++) {
          if (lhsBroadcastedTy.isDynamicDim(i)) {
            transposedOpShapes.push_back(kUnknownSize);
          } else {
            if (lhsBroadcastedShape[i] == rhsBroadcastedShape[i]) {
              transposedOpShapes.push_back(lhsBroadcastedShape[i]);
            } else {
              transposedOpShapes.push_back(lhsBroadcastedShape[i] == 1
                                               ? rhsBroadcastedShape[i]
                                               : lhsBroadcastedShape[i]);
            }
          }
        }
        if (lhsRank > 1)
          transposedOpShapes.push_back(lhsBroadcastedShape[maxInputRank - 2]);
        if (rhsRank > 1)
          transposedOpShapes.push_back(rhsBroadcastedShape[maxInputRank - 1]);

        return;
      };

      SmallVector<int64_t> reshapedOpShape, transposedOpShape;
      SmallVector<int32_t> transposedOpDims;

      computeOpShape(reshapedOpShape, transposedOpDims, transposedOpShape);

      bool opNeedsTranspose = isTransposeRequired(transposedOpDims);

      // Perform reshape
      auto reshapedOpType = RankedTensorType::get(
          makeShapeLLVMCompatible(reshapedOpShape), outputElemTy);
      auto reshapedOp = rewriter.create<rvtensor::ReshapeOp>(
          op->getLoc(),
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              reshapedOpType),
          mmOpResult, rewriter.getDenseI64ArrayAttr(reshapedOpShape));

      if (opNeedsTranspose) {

        std::optional<Value> transposedOpShapeConst =
            rvtensor::getConstTensor<int32_t>(
                rewriter, op,
                /*vec=*/transposedOpDims,
                /*shape=*/{static_cast<int32_t>(transposedOpDims.size())});

        auto transposedOpType = RankedTensorType::get(
            makeShapeLLVMCompatible(transposedOpShape), outputElemTy);
        output = rewriter
                     .create<rvtensor::TransposeOp>(
                         op->getLoc(),
                         OpConversionPattern<AtenOpT>::getTypeConverter()
                             ->convertType(transposedOpType),
                         reshapedOp.getResult(), transposedOpShapeConst.value())
                     .getResult();

      } else {
        output = reshapedOp.getResult();
      }
    } else {
      output = mmOpResult;
    }

    return success();
  }
  // The default version just reads two inputs, computes output and returns it.
  // Other versions may add a bias, apply GEMM-style alpha/beta scaling etc.
  virtual LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value lhs, rhs;

    if (failed(readMatMulInputs(op, adaptor, rewriter, lhs, rhs)))
      return rewriter.notifyMatchFailure(op, "Failed to read matmul inputs");

    Value output;

    if (failed(performMatmul(op, adaptor, rewriter, lhs, rhs, output)))
      return rewriter.notifyMatchFailure(op,
                                         "Failed to perform matmul operation");

    rewriter.replaceOpWithNewOp<tensor::CastOp>(
        op,
        OpConversionPattern<AtenOpT>::getTypeConverter()
            ->convertType(op.getType())
            .template cast<RankedTensorType>(),
        output);

    return success();
  }
};

// Legalizes the torch.matmul op for general n-dim matmul.
template <typename AtenOpT>
class ConvertAtenMatMulOp : public ConvertAtenMatmulBaseOp<AtenOpT> {
public:
  using ConvertAtenMatmulBaseOp<AtenOpT>::ConvertAtenMatmulBaseOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult readMatMulInputs(AtenOpT op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter,
                                 Value &lhs, Value &rhs) const override {
    lhs = adaptor.getSelf();
    auto lhsTy = lhs.getType().cast<RankedTensorType>();

    rhs = adaptor.getOther();
    auto rhsTy = rhs.getType().cast<RankedTensorType>();

    if (!lhsTy || !rhsTy)
      return rewriter.notifyMatchFailure(
          op, "Only ranked tensor types supported in RVTENSOR matmul");

    return success();
  }
};

// Implements handling of aten.mm and aten.bmm ops.
template <typename AtenOpT>
class ConvertAtenMmOp : public ConvertAtenMatmulBaseOp<AtenOpT> {
public:
  using ConvertAtenMatmulBaseOp<AtenOpT>::ConvertAtenMatmulBaseOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult readMatMulInputs(AtenOpT op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter,
                                 Value &lhs, Value &rhs) const override {

    lhs = adaptor.getSelf();
    auto lhsTy = lhs.getType().cast<RankedTensorType>();

    rhs = adaptor.getMat2();
    auto rhsTy = rhs.getType().cast<RankedTensorType>();

    if (!lhsTy || !rhsTy)
      return rewriter.notifyMatchFailure(
          op, "Only ranked tensor types supported in RVTENSOR matmul");

    auto lhsRank = lhsTy.getRank();
    auto rhsRank = rhsTy.getRank();

    if (isa<AtenMmOp>(op)) {
      // Mm takes two 2D tensors.
      if (lhsRank != 2 || rhsRank != 2)
        return op.emitError("aten.mm called but matrix rank != 2");
    } else if (isa<AtenBmmOp>(op)) {
      // Bmm takes two 3D tensors.
      if (lhsRank != 3 || rhsRank != 3)
        return op.emitError("aten.bmm called but matrix rank != 3");
    }

    return success();
  }
};

template <>
LogicalResult ConvertAtenOp<AtenConvolutionOp>::matchAndRewrite(
    AtenConvolutionOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  bool transposed;
  if (!matchPattern(op.getTransposed(), m_TorchConstantBool(&transposed)))
    return rewriter.notifyMatchFailure(
        op, "Unimplemented: non-constant value for transposed not supported");
  if (transposed)
    return rewriter.notifyMatchFailure(
        op, "Unimplemented: transposed convolution not supported");

  auto input = adaptor.getInput();
  auto weight = adaptor.getWeight();

  auto inputTy = input.getType().cast<RankedTensorType>();
  auto weightTy = weight.getType().cast<RankedTensorType>();
  auto outputTy = getTypeConverter()
                      ->convertType(op.getType())
                      .template cast<RankedTensorType>();

  if (!inputTy || !weightTy || !outputTy)
    return rewriter.notifyMatchFailure(
        op, "Input, weight and output to Convolution must be ranked tensors");

  auto inputElemTy = inputTy.getElementType();
  auto weightElemTy = weightTy.getElementType();
  auto inputShape = makeShapeTorchCompatible(inputTy.getShape());
  auto weightShape = makeShapeTorchCompatible(weightTy.getShape());

  if (inputTy.getRank() != 4)
    return rewriter.notifyMatchFailure(
        op, "Unimplemented: only 2D convolutions supported");

  if (!weightTy.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "Unimplemented: RVTENSOR only supports static weight");

  // Bias is optional. RVTENSOR mandates a zero tensor here, so construct one if
  // required.
  auto bias = adaptor.getBias();
  if (adaptor.getBias().getType().template isa<Torch::NoneType>()) {
    // TBD: This is only valid for quantized 8-bit. For 16-bit, the bias (and
    // accumulator) are 48-bit and not 32-bit, and requires the use of APInt to
    // define a 48-bit int.
    if (isa<quant::QuantizedType>(inputElemTy)) {
      SmallVector<int32_t> zeroVec(weightShape[0], 0);
      bias = rvtensor::getConstTensor<int32_t>(
                 rewriter, op, zeroVec, {static_cast<int32_t>(weightShape[0])})
                 .value();
    } else {
      SmallVector<float> zeroVec(weightShape[0], 0);
      bias = rvtensor::getConstTensor<float>(rewriter, op, zeroVec,
                                         {static_cast<int32_t>(weightShape[0])})
                 .value();
    }
  } else {
    if (!bias.getType().cast<RankedTensorType>())
      return rewriter.notifyMatchFailure(
          op, "Bias provided but not a ranked tensor");
  }
  auto biasElemTy =
      isa<mlir::FloatType>(inputElemTy) ? inputElemTy : rewriter.getI32Type();

  int64_t groups;
  if (!matchPattern(op.getGroups(), m_TorchConstantInt(&groups))) {
    return rewriter.notifyMatchFailure(op, "non-const group size unsupported");
  } else if (groups != 1 && weightShape[1] != 1) {
    return rewriter.notifyMatchFailure(
        op, "group size must be 1 (convolution) or weight.dim(1) must be 1 "
            "(depthwise convolution)");
  }

  SmallVector<int64_t, 2> stride;
  if (!matchPattern(adaptor.getStride(), m_TorchListOfConstantInts(stride)))
    return rewriter.notifyMatchFailure(op, "non-const stride list unsupported");

  SmallVector<int64_t, 2> padding_2d;
  if (!matchPattern(adaptor.getPadding(),
                    m_TorchListOfConstantInts(padding_2d)))
    return rewriter.notifyMatchFailure(op,
                                       "non-const padding list unsupported");
  // RVTENSOR uses 4D padding {t, b, l, r} while Torch defines 2D padding {t, l}.
  // The Torch OFM computation uses 2*pad in each spatial direction, implying
  // the same t=b and l=r values for RVTENSOR.
  SmallVector<int64_t> padding(
      {padding_2d[0], padding_2d[0], padding_2d[1], padding_2d[1]});

  SmallVector<int64_t, 2> dilation;
  if (!matchPattern(adaptor.getDilation(), m_TorchListOfConstantInts(dilation)))
    return rewriter.notifyMatchFailure(op,
                                       "non-const dilation list unsupported");

  // RVTENSOR works in NHWC and takes OHWI (conv) / HWIM (depthwise conv) weights.
  // Perform the necessary transformations.
  std::optional<Value> nchwToNhwcTransposeConst =
      rvtensor::getConstTensor<int32_t>(rewriter, op,
                                    /*vec=*/{0, 2, 3, 1},
                                    /*shape=*/{static_cast<int32_t>(4)});
  SmallVector<int64_t> transposedInputShape(
      {inputShape[0], inputShape[2], inputShape[3], inputShape[1]});
  auto transposedInputType = RankedTensorType::get(
      makeShapeLLVMCompatible(transposedInputShape), inputElemTy);
  auto transposedInput =
      rewriter
          .create<rvtensor::TransposeOp>(
              op->getLoc(),
              getTypeConverter()->convertType(transposedInputType), input,
              nchwToNhwcTransposeConst.value())
          .getResult();

  SmallVector<int64_t> transformedWeightShape;
  RankedTensorType transformedWeightType;
  Value transformedWeight;
  int64_t outputCDim;
  if (groups == 1) {
    // full convolution: O(I/G)HW-> OHWI
    transformedWeightShape = {weightShape[0], weightShape[2], weightShape[3],
                              weightShape[1]};
    transformedWeightType = RankedTensorType::get(
        makeShapeLLVMCompatible(transformedWeightShape), weightElemTy);
    transformedWeight =
        rewriter
            .create<rvtensor::TransposeOp>(
                op->getLoc(),
                getTypeConverter()->convertType(transformedWeightType), weight,
                nchwToNhwcTransposeConst.value())
            .getResult();
    outputCDim = transformedWeightShape[0];
  } else if (weightShape[1] == 1) {
    // depthwise convolution: O(I/G)HW-> HWIM)
    // transpose: O(I/G)HW -> HWO(I/G)
    std::optional<Value> transposeConst =
        rvtensor::getConstTensor<int32_t>(rewriter, op,
                                      /*vec=*/{2, 3, 0, 1},
                                      /*shape=*/{static_cast<int32_t>(4)});
    SmallVector<int64_t> transposedWeightShape = {
        weightShape[2], weightShape[3], weightShape[0], weightShape[1]};
    auto transposedWeightType = RankedTensorType::get(
        makeShapeLLVMCompatible(transposedWeightShape), weightElemTy);
    auto transposedWeight =
        rewriter
            .create<rvtensor::TransposeOp>(
                op->getLoc(),
                getTypeConverter()->convertType(transposedWeightType), weight,
                transposeConst.value())
            .getResult();

    // reshape: HWO(I/G) -> HWIM
    outputCDim = makeShapeTorchCompatible(outputTy.getShape())[1];
    if (outputCDim == kUnknownSize) {
      return rewriter.notifyMatchFailure(
          op, "number of output channels must be statically known for "
              "depthwise convolutions");
    }
    transformedWeightShape = {
        transposedWeightShape[0],
        transposedWeightShape[1],
        groups,
        outputCDim / groups,
    };
    transformedWeightType = RankedTensorType::get(
        makeShapeLLVMCompatible(transformedWeightShape), weightElemTy);
    transformedWeight =
        rewriter
            .create<rvtensor::ReshapeOp>(
                op->getLoc(),
                getTypeConverter()->convertType(transformedWeightType),
                transposedWeight,
                rewriter.getDenseI64ArrayAttr(transformedWeightShape))
            .getResult();
  } else {
    llvm_unreachable("Unhandled convolution type");
  }

  int64_t outputHDim, outputWDim;
  if (inputTy.hasStaticShape()) {
    int64_t inputHDim = inputShape[2];
    int64_t inputWDim = inputShape[3];
    int64_t weightHDim = weightShape[2];
    int64_t weightWDim = weightShape[3];
    outputHDim = (inputHDim + padding[0] + padding[1] -
                  dilation[0] * (weightHDim - 1) - 1) /
                     stride[0] +
                 1;
    outputWDim = (inputWDim + padding[2] + padding[3] -
                  dilation[1] * (weightWDim - 1) - 1) /
                     stride[1] +
                 1;
  } else {
    outputHDim = kUnknownSize;
    outputWDim = kUnknownSize;
  }

  // Output shape is NHWC, to be transposed back to NCHW. Output elemTy for
  // quantized input is i32, which gets rescaled down to quantized output range.
  SmallVector<int64_t> outputShape = {transposedInputShape[0], outputHDim,
                                      outputWDim, outputCDim};
  auto convOpTy =
      RankedTensorType::get(makeShapeLLVMCompatible(outputShape), biasElemTy);

  Value convOpResult;
  if (groups == 1) {
    // full convolution
    convOpResult =
        rewriter
            .create<rvtensor::Conv2DOp>(op->getLoc(),
                                    getTypeConverter()->convertType(convOpTy),
                                    transposedInput, transformedWeight, bias,
                                    rewriter.getDenseI64ArrayAttr(padding),
                                    rewriter.getDenseI64ArrayAttr(stride),
                                    rewriter.getDenseI64ArrayAttr(dilation))
            .getResult();
  } else if (weightShape[1] == 1) {
    // depthwise convolution
    convOpResult =
        rewriter
            .create<rvtensor::DepthwiseConv2DOp>(
                op->getLoc(), getTypeConverter()->convertType(convOpTy),
                transposedInput, transformedWeight, bias,
                rewriter.getDenseI64ArrayAttr(padding),
                rewriter.getDenseI64ArrayAttr(stride),
                rewriter.getDenseI64ArrayAttr(dilation))
            .getResult();
  } else {
    llvm_unreachable("Unhandled convolution type");
  }

  std::optional<Value> nhwcToNchwTransposeConst =
      rvtensor::getConstTensor<int32_t>(rewriter, op,
                                    /*vec=*/{0, 3, 1, 2},
                                    /*shape=*/{static_cast<int32_t>(4)});
  SmallVector<int64_t> transposedOutputShape(
      {outputShape[0], outputShape[3], outputShape[1], outputShape[2]});
  auto transposedOutputType = RankedTensorType::get(
      makeShapeLLVMCompatible(transposedOutputShape), biasElemTy);
  auto transposedOutput =
      rewriter
          .create<rvtensor::TransposeOp>(
              op->getLoc(),
              getTypeConverter()->convertType(transposedOutputType),
              convOpResult, nhwcToNchwTransposeConst.value())
          .getResult();

  Value rescaledResult = transposedOutput;
  if (isa<quant::QuantizedType>(inputElemTy)) {
    rescaledResult = rvtensor::buildRescaleOpConvOutput(
        rewriter, op, transposedOutput, inputTy, weightTy, outputTy);
  }

  rewriter.replaceOpWithNewOp<tensor::CastOp>(
      op, getTypeConverter()->convertType(op.getType()), rescaledResult);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenReshapeOp>::matchAndRewrite(
    AtenReshapeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto self = adaptor.getSelf();

  auto selfTy = self.getType().cast<RankedTensorType>();
  if (!selfTy)
    return rewriter.notifyMatchFailure(
        op, "Only ranked tensor types supported in RVTENSOR Reshape");

  // Check that at most one dimension is -1
  SmallVector<int64_t> newShape;
  if (!matchPattern(op.getShape(), m_TorchListOfConstantInts(newShape)))
    return rewriter.notifyMatchFailure(
        op, "Only constant shape supported in RVTENSOR Reshape");

  int auto_sz = 0;
  for (auto s : newShape)
    auto_sz += (s == -1 ? 1 : 0);
  if (auto_sz > 1)
    return rewriter.notifyMatchFailure(
        op, "At most one dimension may be specified as -1 to "
            "automatically calculate its size");

  auto newType = RankedTensorType::get(makeShapeLLVMCompatible(newShape),
                                       selfTy.getElementType());

  rewriter.replaceOpWithNewOp<rvtensor::ReshapeOp>(
      op, getTypeConverter()->convertType(newType), self,
      rewriter.getDenseI64ArrayAttr(newShape));

  return success();
}

// Torch constants are converted to rvtensor.const .
template <>
LogicalResult ConvertAtenOp<ValueTensorLiteralOp>::matchAndRewrite(
    ValueTensorLiteralOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto outputTy = getTypeConverter()
                      ->convertType(op.getType())
                      .template cast<RankedTensorType>();

  // Tensors with integer types need to be converted to signless integer
  // element type. All tensors with element types other than integer can reuse
  // existing elements attribute.
  // TODO: what about unsigned integer?
  if (auto elements = op.getValueAttr().dyn_cast<DenseIntElementsAttr>()) {
    if (elements.getElementType().isSignedInteger()) {
      Type builtinTensorElemTy = outputTy.getElementType();
      unsigned bitWidth = builtinTensorElemTy.getIntOrFloatBitWidth();
      DenseElementsAttr valueAttr =
          elements.mapValues(builtinTensorElemTy, [&](const APInt &v) {
            return APInt(bitWidth, v.getSExtValue());
          });
      rewriter.replaceOpWithNewOp<rvtensor::ConstOp>(op, outputTy, valueAttr);
      return success();
    }
  }
  rewriter.replaceOpWithNewOp<rvtensor::ConstOp>(op, outputTy, adaptor.getValue());
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenTransposeIntOp>::matchAndRewrite(
    AtenTransposeIntOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto selfType = adaptor.getSelf().getType().dyn_cast<TensorType>();
  if (!selfType)
    return rewriter.notifyMatchFailure(op, "Only tensor types are supported");

  // Only statically resolvable values are currently supported
  int64_t dim0, dim1;
  if (!matchPattern(op.getDim0(), m_TorchConstantInt(&dim0)))
    return rewriter.notifyMatchFailure(op, "dim0 must be a Scalar constant");

  if (!matchPattern(op.getDim1(), m_TorchConstantInt(&dim1)))
    return rewriter.notifyMatchFailure(op, "dim1 must be a Scalar constant");

  dim0 = toPositiveDim(dim0, selfType.getRank());
  dim1 = toPositiveDim(dim1, selfType.getRank());

  auto selfRank = selfType.getRank();
  if (!isValidDim(dim0, selfRank) || !isValidDim(dim1, selfRank))
    return rewriter.notifyMatchFailure(
        op, "dim0 and dim1 must be less than tensor rank");

  SmallVector<int32_t> transposeDims;
  for (auto i = 0; i < selfType.getRank(); ++i)
    transposeDims.push_back(i);

  transposeDims[dim0] = dim1;
  transposeDims[dim1] = dim0;

  auto transposeDimsConst = mlir::rvtensor::getConstTensor<int32_t>(
      rewriter, op.getOperation(), transposeDims, {selfType.getRank()});

  rewriter.replaceOpWithNewOp<rvtensor::TransposeOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.getSelf(),
      transposeDimsConst.value());

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenMaxDimOp>::matchAndRewrite(
    AtenMaxDimOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto selfType = adaptor.getSelf().getType().dyn_cast<TensorType>();
  if (!selfType)
    return rewriter.notifyMatchFailure(op, "Only tensor types are supported");

  auto indicesType =
      getTypeConverter()->convertType(op.getType(1)).dyn_cast<TensorType>();
  if (!indicesType)
    return rewriter.notifyMatchFailure(op, "Only tensor types are supported");

  auto selfElemType = selfType.getElementType();
  auto indicesElemType = indicesType.getElementType();

  // Only statically deducible values are currently supported
  int64_t dim;
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
    return rewriter.notifyMatchFailure(op, "dim must be a Scalar constant");

  dim = toPositiveDim(dim, selfType.getRank());

  if (!isValidDim(dim, selfType.getRank()))
    return rewriter.notifyMatchFailure(op, "dim must be less than tensor rank");

  bool keepDim;
  if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepDim)))
    return rewriter.notifyMatchFailure(op, "keepdim must be a Scalar constant");

  SmallVector<int64_t> reducedShape, prunedShape;
  for (auto en :
       llvm::enumerate(makeShapeTorchCompatible(selfType.getShape()))) {
    if (static_cast<int64_t>(en.index()) == dim) {
      reducedShape.push_back(1);
      continue;
    }
    reducedShape.push_back(en.value());
    prunedShape.push_back(en.value());
  }

  auto dimAttr = rewriter.getIntegerAttr(rewriter.getI32Type(), dim);
  auto prunedShapeAttr = rewriter.getDenseI64ArrayAttr(prunedShape);

  Value reduceMax = rewriter.create<rvtensor::ReduceMaxOp>(
      op->getLoc(),
      RankedTensorType::get(makeShapeLLVMCompatible(reducedShape),
                            selfElemType),
      adaptor.getSelf(), dimAttr);

  Value argMax = rewriter.create<rvtensor::ArgMaxOp>(
      op->getLoc(),
      RankedTensorType::get(makeShapeLLVMCompatible(prunedShape),
                            indicesElemType),
      adaptor.getSelf(), dimAttr);

  if (argMax.getType() != indicesType) {
    argMax = rewriter.create<rvtensor::ReshapeOp>(
        op->getLoc(), indicesType, argMax,
        rewriter.getDenseI64ArrayAttr(reducedShape));
  }

  if (!keepDim) {
    reduceMax = rewriter.create<rvtensor::ReshapeOp>(
        op->getLoc(),
        RankedTensorType::get(makeShapeLLVMCompatible(prunedShape),
                              selfElemType),
        reduceMax, prunedShapeAttr);
  }

  rewriter.replaceOp(op, {reduceMax, argMax});

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenSliceTensorOp>::matchAndRewrite(
    AtenSliceTensorOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto selfType = adaptor.getSelf().getType().dyn_cast<TensorType>();
  if (!selfType || !selfType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "Only tensor types with static shape are supported");

  // Only statically deducible values are currently supported
  int64_t dim;
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
    return rewriter.notifyMatchFailure(op, "dim must be a Scalar constant");

  dim = toPositiveDim(dim, selfType.getRank());

  if (!isValidDim(dim, selfType.getRank()))
    return rewriter.notifyMatchFailure(op, "dim must less than tensor rank");

  int64_t start;
  if (!matchPattern(op.getStart(), m_TorchConstantInt(&start)))
    return rewriter.notifyMatchFailure(op, "start must be a Scalar constant");

  if (start < 0) {
    start = toPositiveDim(start, selfType.getShape()[dim]);
    if (!isValidDim(start, selfType.getShape()[dim]))
      return rewriter.notifyMatchFailure(op, "start is not a valid index");
  }
  start = std::min(selfType.getShape()[dim], start);

  int64_t end;
  if (!matchPattern(op.getEnd(), m_TorchConstantInt(&end))) {
    if (isa<ConstantNoneOp>(op.getEnd().getDefiningOp()))
      end = selfType.getShape()[dim];
    else
      return rewriter.notifyMatchFailure(op, "end must be a Scalar constant");
  }
  // support for end < 0
  end = toPositiveDim(end, selfType.getShape()[dim]);
  // support for end out of upper bound
  end = (end > selfType.getShape()[dim] ? selfType.getShape()[dim] : end);

  // FIXME: add support for start < 0 and end < start
  if (end < start)
    return rewriter.notifyMatchFailure(op,
                                       "Currently unsupported: end < start");

  int64_t step;
  if (!matchPattern(op.getStep(), m_TorchConstantInt(&step)))
    return rewriter.notifyMatchFailure(op, "step must be a Scalar constant");

  if (step != 1)
    return rewriter.notifyMatchFailure(
        op, "step value other than 1 is currently unsupported");

  SmallVector<int64_t> startSlice(selfType.getRank(), 0);
  SmallVector<int64_t> sizeSlice =
      llvm::to_vector(makeShapeTorchCompatible(selfType.getShape()));

  startSlice[dim] = start;
  sizeSlice[dim] = end - start;

  rewriter.replaceOpWithNewOp<rvtensor::SliceOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.getSelf(),
      rewriter.getDenseI64ArrayAttr(startSlice),
      rewriter.getDenseI64ArrayAttr(sizeSlice));

  return success();
}

template <>
LogicalResult ConvertAtenOp<PrimNumToTensorScalarOp>::matchAndRewrite(
    PrimNumToTensorScalarOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  const TypeConverter *typeConverter = this->getTypeConverter();
  RankedTensorType resultType =
      typeConverter->convertType(op->getResult(0).getType())
          .cast<RankedTensorType>();

  // Only supports integer operand type, because for the floating point operand
  // type result tensor has to be of type `f64` which is not supported in the
  // rvtensor.
  double doubleValue;
  auto isDouble = matchPattern(op.getA(), m_TorchConstantFloat(&doubleValue));
  int64_t intValue;
  auto isInt = matchPattern(op.getA(), m_TorchConstantInt(&intValue));
  if (!isDouble && !isInt)
    return rewriter.notifyMatchFailure(op,
                                       "Unable to extract the scalar constant");

  auto outElemTy = resultType.getElementType();
  if (isa<mlir::IntegerType>(outElemTy)) {
    rewriter.replaceOpWithNewOp<rvtensor::ConstOp>(
        op, resultType, DenseElementsAttr::get(resultType, {intValue}));
  } else if (outElemTy.isF64()) {
    rewriter.replaceOpWithNewOp<rvtensor::ConstOp>(
        op, resultType, DenseElementsAttr::get(resultType, {doubleValue}));
  }

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenCopyOp>::matchAndRewrite(
    AtenCopyOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = adaptor.getSelf().getType().dyn_cast<TensorType>();
  auto srcType = adaptor.getSrc().getType().dyn_cast<TensorType>();
  if (!selfType || !selfType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "Only tensor types with static shape are supported");

  if (!srcType || !srcType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "Only tensor types with static shape are supported");

  // The non_blocking should be a constant `False`.
  bool nonBlocking;
  if (!matchPattern(op.getNonBlocking(), m_TorchConstantBool(&nonBlocking))) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: non_blocking must be a constant");
  } else if (nonBlocking) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: non_blocking is expected to be false");
  }

  SmallVector<int64_t> selfShape(makeShapeTorchCompatible(selfType.getShape()));
  SmallVector<int64_t> srcShape(makeShapeTorchCompatible(srcType.getShape()));

  if (llvm::equal(selfShape, srcShape) || selfShape.size() == 0) {
    // If we reach here, then it means the given case is handled by implicit
    // broadcasting done by rvtensor.
    Value result;
    if (failed(rvtensor::rvtensorCastTensorToType(
            rewriter, op, adaptor.getSrc(),
            getTypeConverter()->convertType(op.getType()), result)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: cast to result type not supported");
    rewriter.replaceOp(op, result);
    return success();
  }
  return rewriter.notifyMatchFailure(
      op, "unimplemented: valsem.aten.copy op not supported for this case.");
}

//  Legalizes the torch.aten.to.dtype op
template <>
LogicalResult ConvertAtenOp<AtenToDtypeOp>::matchAndRewrite(
    AtenToDtypeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = adaptor.getSelf().getType().dyn_cast<TensorType>();
  if (!selfType || !selfType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "Only tensor types with static shape are supported");

  // The non_blocking arg should be a constant `False`.
  bool nonBlocking;
  if (!matchPattern(op.getNonBlocking(), m_TorchConstantBool(&nonBlocking))) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: non_blocking arg must be a constant");
  } else if (nonBlocking) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: non_blocking arg is expected to be false");
  }

  // The copy arg should be a constant `False`.
  bool copy;
  if (!matchPattern(op.getCopy(), m_TorchConstantBool(&copy))) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: copy arg must be a constant");
  } else if (copy) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: copy arg is expected to be false");
  }

  // Only `none`, `contiguous` and `preserve` memory_format is supported.
  if (!op.getMemoryFormat().getType().isa<Torch::NoneType>()) {
    int64_t memoryFormat;
    if (!matchPattern(op.getMemoryFormat(), m_TorchConstantInt(&memoryFormat)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: the memory format should be specified in "
              "an integer constant");
    if (memoryFormat != torch_upstream::MemoryFormat::Contiguous &&
        memoryFormat != torch_upstream::MemoryFormat::Preserve)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only none, contiguous and preserve "
              "memory_format is supported");
  }

  auto resultTy = getTypeConverter()
                      ->convertType(op.getResult().getType())
                      .cast<RankedTensorType>();

  Value result;
  if (failed(rvtensor::rvtensorCastTensorToType(rewriter, op, adaptor.getSelf(),
                                        resultTy, result)))
    return rewriter.notifyMatchFailure(op, "conversion to result type failed");

  rewriter.replaceOp(op, result);
  return success();
}

// Ref: Error checking based on the Torch to LinAlg lowering
template <typename AtenOpT, int fillVal>
class ConvertAtenConstPatternOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                       ->convertType(op.getType())
                       .template dyn_cast<TensorType>();

    if (!outType)
      return rewriter.notifyMatchFailure(op,
                                         "Only Tensor types supported in RVTENSOR");

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "Only floating-point or integer datatype legalization supported");

    // FIXME: Handle layout, device and pin_memory. Assume dtype has been
    // processed to set output type correctly?
    // The layout arg should be either `none` or `0` i.e. strided.
    if (!op.getLayout().getType().template isa<Torch::NoneType>()) {
      int64_t tensorLayout;
      if (!matchPattern(op.getLayout(), m_TorchConstantInt(&tensorLayout)))
        return rewriter.notifyMatchFailure(
            op, "The layout arg should be either `none` or `0` i.e. strided.");
      else if (tensorLayout != torch_upstream::Layout::Strided)
        return rewriter.notifyMatchFailure(
            op, "The layout arg should be either `none` or `0` i.e. strided.");
    }

    bool pinMemory;
    if (!op.getPinMemory().getType().template isa<Torch::NoneType>() &&
        (!matchPattern(op.getPinMemory(), m_TorchConstantBool(&pinMemory)) ||
         pinMemory)) {
      return rewriter.notifyMatchFailure(
          op, "Unsupported pin_memory, should be either None or false");
    }

    SmallVector<int64_t> shape;
    if (!matchPattern(op.getSize(), m_TorchListOfConstantInts(shape))) {
      return rewriter.notifyMatchFailure(
          op, "Shape must be a list of Scalar constants");
    }

    int64_t size = 1;
    for (auto s : shape)
      size *= s;

    SmallVector<int32_t> values(size, fillVal);
    auto constOp =
        rvtensor::getConstTensor<int32_t>(rewriter, op, values, shape).value();

    rewriter.replaceOpWithNewOp<rvtensor::CastOp>(op, outType, constOp);

    return success();
  }
};

template <typename AtenOpT>
class ConvertAtenFillScalarOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                       ->convertType(op.getType())
                       .template dyn_cast<TensorType>();

    if (!outType || !outType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "Only Tensor types with static shapes are currently supported");

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat()) {
      return rewriter.notifyMatchFailure(
          op, "Only floating-point or integer datatype legalization supported");
    }
    Value constOp;
    if (failed(torchScalarToRVTensorTensor(
            rewriter, op, op.getValue(), constOp, outElemTy,
            makeShapeTorchCompatible(outType.getShape()))))
      return rewriter.notifyMatchFailure(
          op, "Supplied value must be a Scalar constant");

    rewriter.replaceOpWithNewOp<rvtensor::CastOp>(op, outType, constOp);

    return success();
  }
};

template <typename AtenOpT>
class ConvertAtenMaskedFillOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                       ->convertType(op.getType())
                       .template dyn_cast<TensorType>();

    if (!outType || !outType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "Only Tensor types with static shapes are currently supported");

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat()) {
      return rewriter.notifyMatchFailure(
          op, "Only floating-point or integer datatype legalization supported");
    }

    // Not a tensor type.
    auto selfType = adaptor.getSelf().getType().template dyn_cast<TensorType>();
    if (!selfType || !outType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op,
          "Only tensor types with static shapes input are currently supported");

    auto maskType = adaptor.getMask().getType().template dyn_cast<TensorType>();
    if (!maskType)
      return rewriter.notifyMatchFailure(
          op, "Only tensor types mask are currently supported");

    Value rhs = adaptor.getValue();
    auto rhsType = rhs.getType().template dyn_cast<TensorType>();
    Value rhsAsTensor;
    if (!rhsType) { // scalar
      if (failed(torchScalarToRVTensorTensor(rewriter, op, op.getValue(),
                                         rhsAsTensor, rhs.getType(), {})))
        return rewriter.notifyMatchFailure(
            op, "Currently only scalar constants are supported for "
                "conversion in RVTENSOR operation");
    } else { // tensor
      rhsType = rhs.getType().dyn_cast<TensorType>();
    }

    auto rhsTensor = rhsType ? rhs : rhsAsTensor;
    auto rhsTensorType = rhsTensor.getType().template dyn_cast<TensorType>();
    if (rhsTensorType.getElementType() != outElemTy)
      rhsTensor = rewriter.create<rvtensor::CastOp>(
          op.getLoc(),
          RankedTensorType::get(rhsTensorType.getShape(), outElemTy),
          rhsTensor);

    rewriter.replaceOpWithNewOp<rvtensor::SelectOp>(op, outType, adaptor.getMask(),
                                                rhsTensor, adaptor.getSelf());
    return success();
  }
};

template <>
LogicalResult ConvertAtenOp<AtenConstantPadNdOp>::matchAndRewrite(
    AtenConstantPadNdOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value self = adaptor.getSelf();
  auto selfTy = self.getType().cast<RankedTensorType>();
  auto selfElemTy = selfTy.getElementType();
  int64_t rank = selfTy.getRank();

  // START the code snippet from
  // lib/Conversion/TorchToLinalg/TensorConstructors.cpp (see:
  // ConvertAtenConstantPadNdOp) Pattern match against the op's original
  // operands, because otherwise we will get the lowered version of the operands
  // which is harder to pattern match.
  SmallVector<int64_t> padInts;
  if (!matchPattern(op.getPad(), m_TorchListOfConstantInts(padInts)))
    return rewriter.notifyMatchFailure(op,
                                       "only support constant int pad ranges");
  uint64_t padRank = padInts.size() / 2;
  if (padRank * 2 != padInts.size())
    return rewriter.notifyMatchFailure(op, "pad range size is not even");
  if (rank < 0 || padRank > (uint64_t)rank)
    return rewriter.notifyMatchFailure(op, "padding exceeds tensor rank");

  // Initialize low/high paddings with 0 for all the dims.
  SmallVector<int64_t> lowPadding(/*Size=*/rank, /*Value=*/0);
  SmallVector<int64_t> highPadding(/*Size=*/rank, /*Value=*/0);
  // Add the requested padding - note op.pad() is highest dim first ordered
  // pairs of low,high.
  for (uint64_t i = 0; i < padRank; ++i) {
    lowPadding[rank - i - 1] = padInts[i * 2];
    highPadding[rank - i - 1] = padInts[i * 2 + 1];
  }
  // END the code snippet from
  // lib/Conversion/TorchToLinalg/TensorConstructors.cpp (see:
  // ConvertAtenConstantPadNdOp)

  llvm::SmallVector<int64_t> translatePadsList;

  for (unsigned int i = 0; i < rank; i++) {
    translatePadsList.push_back(lowPadding[i]);
    translatePadsList.push_back(highPadding[i]);
  }

  DenseElementsAttr paddingAttr = DenseIntElementsAttr::get(
      RankedTensorType::get({rank, 2}, rewriter.getI64Type()),
      translatePadsList);

  Value padsList1 = rewriter.create<mlir::rvtensor::ConstOp>(
      loc, paddingAttr.getType(), paddingAttr);

  Value padValue = adaptor.getValue();
  Operation *padOp = padValue.getDefiningOp();
  padValue = padOp->getOperand(0);

  Value padTensor;
  if (failed(torchScalarToRVTensorTensor(rewriter, op.getOperation(), padValue,
                                     padTensor, selfElemTy, {})))
    return rewriter.notifyMatchFailure(
        op, "Pad value needs to be a scalar constant for conversion to "
            "RVTENSOR pad operation");

  rewriter.replaceOpWithNewOp<mlir::rvtensor::PadOp>(
      op, getTypeConverter()->convertType(op.getType()), self, padsList1,
      padTensor);
  return success();
}

} // namespace

// -----------------------------------------------------------------------------
// TorchToRVTensor Pass
// -----------------------------------------------------------------------------

namespace {
class ConvertTorchToRVTensor : public ConvertTorchToRVTensorBase<ConvertTorchToRVTensor> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<rvtensor::RVTensorDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<rvtensor::RVTensorDialect, tensor::TensorDialect,
                           arith::ArithDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    // When we have a chain of torch.constant.int followed by a unsupported
    // torch op, we want the pass to mention the unsupported torch op
    // in the error message.
    target.addLegalOp<ConstantNoneOp>();
    target.addLegalOp<ConstantBoolOp>();
    target.addLegalOp<ConstantIntOp>();
    target.addLegalOp<ConstantFloatOp>();
    target.addLegalOp<ConstantStrOp>();
    target.addLegalOp<ConstantDeviceOp>();
    target.addLegalOp<PrimListConstructOp>();
    target.addLegalOp<PrimTupleConstructOp>();
    target.addIllegalDialect<Torch::TorchDialect>();

    RewritePatternSet patterns(context);

#define INSERT_MATMUL_ATENOP_PATTERN(AtenOp)                                   \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenMatMulOp<AtenOp>>(typeConverter, context);
    INSERT_MATMUL_ATENOP_PATTERN(AtenMatmulOp);
#undef INSERT_MATMUL_ATEMOP_PATTERN

#define INSERT_MM_ATENOP_PATTERN(AtenOp)                                       \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenMmOp<AtenOp>>(typeConverter, context);
    INSERT_MM_ATENOP_PATTERN(AtenMmOp);
    INSERT_MM_ATENOP_PATTERN(AtenBmmOp);
#undef INSERT_MM_ATEMOP_PATTERN

#define INSERT_CONSTANT_FILL_PATTERN(AtenOp, fillVal)                          \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenConstPatternOp<AtenOp, fillVal>>(typeConverter,      \
                                                           context);
    INSERT_CONSTANT_FILL_PATTERN(AtenOnesOp, 1);
    INSERT_CONSTANT_FILL_PATTERN(AtenZerosOp, 0);
#undef INSERT_CONSTANT_FILL_PATTERN

#define INSERT_FILL_SCALAR_PATTERN(AtenOp)                                     \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenFillScalarOp<AtenOp>>(typeConverter, context);
    INSERT_FILL_SCALAR_PATTERN(AtenFill_ScalarOp);
#undef INSERT_FILL_SCALAR_PATTERN

#define INSERT_MASKED_FILL_PATTERN(AtenOp)                                     \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenMaskedFillOp<AtenOp>>(typeConverter, context);
    INSERT_MASKED_FILL_PATTERN(AtenMaskedFillScalarOp);
    INSERT_MASKED_FILL_PATTERN(AtenMaskedFillTensorOp);
#undef INSERT_MASKED_FILL_PATTERN

#define INSERT_ATENOP_PATTERN(AtenOp)                                          \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenOp<AtenOp>>(typeConverter, context);
    INSERT_ATENOP_PATTERN(AtenConvolutionOp);
    INSERT_ATENOP_PATTERN(AtenReshapeOp);
    INSERT_ATENOP_PATTERN(AtenTransposeIntOp);
#undef INSERT_ATENOP_PATTERN

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::createConvertTorchToRVTensorPass() {
  return std::make_unique<ConvertTorchToRVTensor>();
}
