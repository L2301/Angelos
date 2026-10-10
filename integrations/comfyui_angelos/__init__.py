"""Manually installed, local-only Angelos export nodes for ComfyUI."""

from .nodes import AngelosAudience, AngelosCampaign, AngelosExportCreative
from .upl import AngelosChoosePrompt, AngelosParseUPL, AngelosPromptTemplate
from .upl import AngelosScalarCompare, AngelosScalarMap, AngelosScalarToInt
from .upl import AngelosUPLFactFloat, AngelosUPLFactString, AngelosUPLVectorFloat
from .upl import AngelosUPLEmbeddingBlock, AngelosUPLRecentIntentText


NODE_CLASS_MAPPINGS = {
    "AngelosAudience": AngelosAudience,
    "AngelosCampaign": AngelosCampaign,
    "AngelosExportCreative": AngelosExportCreative,
    "AngelosParseUPL": AngelosParseUPL,
    "AngelosUPLFactString": AngelosUPLFactString,
    "AngelosUPLFactFloat": AngelosUPLFactFloat,
    "AngelosUPLVectorFloat": AngelosUPLVectorFloat,
    "AngelosUPLEmbeddingBlock": AngelosUPLEmbeddingBlock,
    "AngelosUPLRecentIntentText": AngelosUPLRecentIntentText,
    "AngelosScalarToInt": AngelosScalarToInt,
    "AngelosScalarMap": AngelosScalarMap,
    "AngelosScalarCompare": AngelosScalarCompare,
    "AngelosPromptTemplate": AngelosPromptTemplate,
    "AngelosChoosePrompt": AngelosChoosePrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AngelosAudience": "Angelos Audience (Synthetic Adults)",
    "AngelosCampaign": "Angelos Campaign",
    "AngelosExportCreative": "Angelos Export Creative (HD Teacher Pack)",
    "AngelosParseUPL": "Angelos Parse UPL JSON",
    "AngelosUPLFactString": "Angelos UPL Fact → STRING",
    "AngelosUPLFactFloat": "Angelos UPL Fact → FLOAT",
    "AngelosUPLVectorFloat": "Angelos UPL Coordinate → FLOAT (Opaque)",
    "AngelosUPLEmbeddingBlock": "Angelos UPL Opaque Embedding Bundle (Not CLIP)",
    "AngelosUPLRecentIntentText": "Angelos UPL Recent Intent → STRING",
    "AngelosScalarToInt": "Angelos Scalar → INT (Explicit Mapping)",
    "AngelosScalarMap": "Angelos Scalar → FLOAT (Explicit Mapping)",
    "AngelosScalarCompare": "Angelos Scalar → BOOLEAN (Compare)",
    "AngelosPromptTemplate": "Angelos Safe Prompt Template",
    "AngelosChoosePrompt": "Angelos Choose Prompt",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
