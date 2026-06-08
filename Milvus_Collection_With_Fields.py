from pymilvus import MilvusClient, DataType, Function, FunctionType

client = MilvusClient(uri="http://localhost:19530")

# 1. Build schema
schema = MilvusClient.create_schema(
    auto_id=True,
    enable_dynamic_field=True,
)

# Primary key
schema.add_field("id", DataType.INT64, is_primary=True)

# Raw text field — BM25 reads from here
schema.add_field(
    "text",
    DataType.VARCHAR,
    max_length=65535,
    enable_analyzer=True,      # required for BM25 tokenisation
)

# Dense vector — OpenAI text-embedding-3-small
schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=1536)

# Sparse vector — BM25 writes to here (do NOT manually insert values)
schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

# Optional metadata
schema.add_field("source", DataType.VARCHAR, max_length=512)

# 2. Attach the BM25 Function — this is what unlocks the BM25 metric
bm25_fn = Function(
    name="bm25_fn",
    function_type=FunctionType.BM25,
    input_field_names=["text"],         # VARCHAR field to tokenise
    output_field_names=["sparse_vector"],  # sparse field to populate
)
schema.add_function(bm25_fn)

# 3. Build index params
index_params = client.prepare_index_params()

# Dense HNSW index
index_params.add_index(
    field_name="dense_vector",
    index_name="dense_index",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 32, "efConstruction": 200},
)

# Sparse BM25 index — metric_type BM25 is now valid because the Function is attached
index_params.add_index(
    field_name="sparse_vector",
    index_name="sparse_index",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="BM25",                 # only works because of bm25_fn above
    params={
        "bm25_k1": 1.2,   # term frequency saturation (default 1.2)
        "bm25_b":  0.75,  # document length normalisation (default 0.75)
    },
)

# 4. Create the collection
client.create_collection(
    collection_name="hybrid_rag_docs",
    schema=schema,
    index_params=index_params,
)

print("Collection created. Verify in Attu — sparse_vector should show BM25 metric.")