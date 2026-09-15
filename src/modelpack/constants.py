"""Media types and annotations defined by the CNCF ModelPack specification."""

OCI_MANIFEST_MEDIA_TYPE = "application/vnd.oci.image.manifest.v1+json"
MODEL_MANIFEST_ARTIFACT_TYPE = "application/vnd.cncf.model.manifest.v1+json"
MODEL_CONFIG_MEDIA_TYPE = "application/vnd.cncf.model.config.v1+json"

WEIGHT_RAW_MEDIA_TYPE = "application/vnd.cncf.model.weight.v1.raw"
WEIGHT_TAR_MEDIA_TYPE = "application/vnd.cncf.model.weight.v1.tar"
WEIGHT_TAR_GZIP_MEDIA_TYPE = "application/vnd.cncf.model.weight.v1.tar+gzip"
WEIGHT_TAR_ZSTD_MEDIA_TYPE = "application/vnd.cncf.model.weight.v1.tar+zstd"

WEIGHT_CONFIG_RAW_MEDIA_TYPE = "application/vnd.cncf.model.weight.config.v1.raw"
WEIGHT_CONFIG_TAR_MEDIA_TYPE = "application/vnd.cncf.model.weight.config.v1.tar"
WEIGHT_CONFIG_TAR_GZIP_MEDIA_TYPE = (
    "application/vnd.cncf.model.weight.config.v1.tar+gzip"
)
WEIGHT_CONFIG_TAR_ZSTD_MEDIA_TYPE = (
    "application/vnd.cncf.model.weight.config.v1.tar+zstd"
)

DOC_RAW_MEDIA_TYPE = "application/vnd.cncf.model.doc.v1.raw"
DOC_TAR_MEDIA_TYPE = "application/vnd.cncf.model.doc.v1.tar"
DOC_TAR_GZIP_MEDIA_TYPE = "application/vnd.cncf.model.doc.v1.tar+gzip"
DOC_TAR_ZSTD_MEDIA_TYPE = "application/vnd.cncf.model.doc.v1.tar+zstd"

CODE_RAW_MEDIA_TYPE = "application/vnd.cncf.model.code.v1.raw"
CODE_TAR_MEDIA_TYPE = "application/vnd.cncf.model.code.v1.tar"
CODE_TAR_GZIP_MEDIA_TYPE = "application/vnd.cncf.model.code.v1.tar+gzip"
CODE_TAR_ZSTD_MEDIA_TYPE = "application/vnd.cncf.model.code.v1.tar+zstd"

DATASET_RAW_MEDIA_TYPE = "application/vnd.cncf.model.dataset.v1.raw"
DATASET_TAR_MEDIA_TYPE = "application/vnd.cncf.model.dataset.v1.tar"
DATASET_TAR_GZIP_MEDIA_TYPE = "application/vnd.cncf.model.dataset.v1.tar+gzip"
DATASET_TAR_ZSTD_MEDIA_TYPE = "application/vnd.cncf.model.dataset.v1.tar+zstd"

FILEPATH_ANNOTATION = "org.cncf.model.filepath"
FILE_METADATA_ANNOTATION = "org.cncf.model.file.metadata+json"
UNTESTED_MEDIA_TYPE_ANNOTATION = "org.cncf.model.file.mediatype.untested"
OCI_TITLE_ANNOTATION = "org.opencontainers.image.title"

MODEL_LAYER_MEDIA_TYPES = frozenset(
    {
        WEIGHT_RAW_MEDIA_TYPE,
        WEIGHT_TAR_MEDIA_TYPE,
        WEIGHT_TAR_GZIP_MEDIA_TYPE,
        WEIGHT_TAR_ZSTD_MEDIA_TYPE,
        WEIGHT_CONFIG_RAW_MEDIA_TYPE,
        WEIGHT_CONFIG_TAR_MEDIA_TYPE,
        WEIGHT_CONFIG_TAR_GZIP_MEDIA_TYPE,
        WEIGHT_CONFIG_TAR_ZSTD_MEDIA_TYPE,
        DOC_RAW_MEDIA_TYPE,
        DOC_TAR_MEDIA_TYPE,
        DOC_TAR_GZIP_MEDIA_TYPE,
        DOC_TAR_ZSTD_MEDIA_TYPE,
        CODE_RAW_MEDIA_TYPE,
        CODE_TAR_MEDIA_TYPE,
        CODE_TAR_GZIP_MEDIA_TYPE,
        CODE_TAR_ZSTD_MEDIA_TYPE,
        DATASET_RAW_MEDIA_TYPE,
        DATASET_TAR_MEDIA_TYPE,
        DATASET_TAR_GZIP_MEDIA_TYPE,
        DATASET_TAR_ZSTD_MEDIA_TYPE,
    }
)

TAR_MEDIA_TYPES = frozenset(
    {
        WEIGHT_TAR_MEDIA_TYPE,
        WEIGHT_TAR_GZIP_MEDIA_TYPE,
        WEIGHT_TAR_ZSTD_MEDIA_TYPE,
        WEIGHT_CONFIG_TAR_MEDIA_TYPE,
        WEIGHT_CONFIG_TAR_GZIP_MEDIA_TYPE,
        WEIGHT_CONFIG_TAR_ZSTD_MEDIA_TYPE,
        DOC_TAR_MEDIA_TYPE,
        DOC_TAR_GZIP_MEDIA_TYPE,
        DOC_TAR_ZSTD_MEDIA_TYPE,
        CODE_TAR_MEDIA_TYPE,
        CODE_TAR_GZIP_MEDIA_TYPE,
        CODE_TAR_ZSTD_MEDIA_TYPE,
        DATASET_TAR_MEDIA_TYPE,
        DATASET_TAR_GZIP_MEDIA_TYPE,
        DATASET_TAR_ZSTD_MEDIA_TYPE,
    }
)
