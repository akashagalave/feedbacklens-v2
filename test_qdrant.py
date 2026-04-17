from qdrant_client import QdrantClient

client = QdrantClient(
    url="https://adc13c31-71c6-4ff9-a316-a2d8a85d7889.us-east-1-1.aws.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6ZjM0NTc2ZjgtNjgxMi00MGRjLThhNDMtYWFkZmY0ZTRhNTc4In0.uMeRX_zsB9ouYAU9v2KDslpXcsolKLmbZAPir-FXTsQ"
)

collections = client.get_collections()
print("Collections:", collections)

for col in collections.collections:
    info = client.get_collection(col.name)
    print(f"Collection: {col.name}, vectors: {info.vectors_count}")