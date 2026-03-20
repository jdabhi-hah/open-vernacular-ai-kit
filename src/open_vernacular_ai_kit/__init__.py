"""Open Vernacular AI Kit.

Public API is intentionally small. Prefer using `CodeMixPipeline` + `CodeMixConfig` for SDK usage.
"""

from .app_flows import (
    BatchProcessSummary,
    WhatsAppMessage,
    clean_whatsapp_chat_text,
    parse_whatsapp_export,
    process_csv_batch,
    process_jsonl_batch,
)
from .codemix_render import (
    analyze_codemix,
    analyze_codemix_with_config,
    render_codemix,
    render_codemix_with_config,
)
from .codeswitch import CodeSwitchMetrics, compute_code_switch_metrics
from .config import CodeMixConfig
from .dialect_datasets import (
    DialectIdExample,
    DialectNormalizationExample,
    dump_dialect_id_jsonl,
    dump_dialect_normalization_jsonl,
    load_dialect_id_jsonl,
    load_dialect_normalization_jsonl,
)
from .dialects import (
    DialectDetection,
    DialectNormalizationResult,
    GujaratiDialect,
    detect_dialect,
    detect_dialect_from_tagged_tokens,
    detect_dialect_from_tokens,
    normalize_dialect_tokens,
)
from .errors import (
    DownloadError,
    GckError,
    IntegrationError,
    InvalidConfigError,
    OfflinePolicyError,
    OptionalDependencyError,
)
from .normalize import normalize_text
from .pipeline import CodeMixPipeline, CodeMixPipelineResult
from .rag import RagDocument, RagIndex, RagQuery, RagSearchResult, make_hf_embedder
from .rag_datasets import (
    RagDataset,
    download_gujarat_facts_dataset,
    download_vernacular_facts_dataset,
    load_gujarat_facts_tiny,
    load_vernacular_facts_tiny,
)
from .sarvam_review import (
    SarvamTeacherReviewedRecord,
    dump_reviewed_records_jsonl,
    init_review_record,
    init_review_records_from_candidates,
    load_reviewed_records_jsonl,
)
from .sarvam_seed import (
    SarvamTeacherSeedRecord,
    build_failure_seed,
    build_failure_seed_from_golden_translit,
    build_failure_seed_from_language_sentences,
    dump_teacher_seed_jsonl,
)
from .sarvam_teacher import (
    SarvamTeacherCandidateRecord,
    SarvamTeacherInput,
    SarvamTeacherTokenCandidate,
    build_sarvam_teacher_prompt,
    dump_sarvam_teacher_records_jsonl,
    load_sarvam_teacher_inputs_jsonl,
    mine_sarvam_teacher_candidate,
    parse_sarvam_teacher_response,
)

__all__ = [
    "CodeMixConfig",
    "CodeMixPipeline",
    "CodeMixPipelineResult",
    "analyze_codemix",
    "analyze_codemix_with_config",
    "CodeSwitchMetrics",
    "compute_code_switch_metrics",
    "DialectDetection",
    "DialectNormalizationResult",
    "GujaratiDialect",
    "detect_dialect",
    "detect_dialect_from_tokens",
    "detect_dialect_from_tagged_tokens",
    "normalize_dialect_tokens",
    "DialectIdExample",
    "DialectNormalizationExample",
    "load_dialect_id_jsonl",
    "load_dialect_normalization_jsonl",
    "dump_dialect_id_jsonl",
    "dump_dialect_normalization_jsonl",
    "normalize_text",
    "render_codemix",
    "render_codemix_with_config",
    "BatchProcessSummary",
    "WhatsAppMessage",
    "clean_whatsapp_chat_text",
    "parse_whatsapp_export",
    "process_csv_batch",
    "process_jsonl_batch",
    "GckError",
    "InvalidConfigError",
    "OptionalDependencyError",
    "OfflinePolicyError",
    "DownloadError",
    "IntegrationError",
    "RagDocument",
    "RagQuery",
    "RagSearchResult",
    "RagIndex",
    "make_hf_embedder",
    "RagDataset",
    "load_vernacular_facts_tiny",
    "download_vernacular_facts_dataset",
    "load_gujarat_facts_tiny",
    "download_gujarat_facts_dataset",
    "SarvamTeacherInput",
    "SarvamTeacherTokenCandidate",
    "SarvamTeacherCandidateRecord",
    "SarvamTeacherReviewedRecord",
    "SarvamTeacherSeedRecord",
    "build_sarvam_teacher_prompt",
    "build_failure_seed",
    "build_failure_seed_from_language_sentences",
    "build_failure_seed_from_golden_translit",
    "parse_sarvam_teacher_response",
    "mine_sarvam_teacher_candidate",
    "load_sarvam_teacher_inputs_jsonl",
    "dump_sarvam_teacher_records_jsonl",
    "dump_teacher_seed_jsonl",
    "init_review_record",
    "init_review_records_from_candidates",
    "load_reviewed_records_jsonl",
    "dump_reviewed_records_jsonl",
]

__version__ = "1.2.0rc1"
 
