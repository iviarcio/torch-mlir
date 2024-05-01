add_mlir_conversion_library(TorchMLIRTorchToRVTensor
    TorchToRVTensor.cpp
    RVTLegalizeUtils.cpp
    RVTLegalizeCommon.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/TorchToRVTensor

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  RVTensorDialect
  TorchMLIRConversionUtils
  TorchMLIRTorchDialect
)

torch_mlir_target_includes(TorchMLIRTorchToRVTensor)
