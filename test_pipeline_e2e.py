import asyncio
import os
import sys

# Append the current directory so app imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

from backend.app.services.pipeline_service import PipelineService

async def main():
    pipeline = PipelineService()
    
    # Read the tree and leaf images into bytes
    try:
        with open("test_tree_integration.jpg", "rb") as f:
            tree_bytes = f.read()
            
        leaf_path = "module1_species_classification/tree_species_classifier/data/processed_pilot/Mango/Mango_000000.jpg"
        if not os.path.exists(leaf_path):
            # Try to grab another known image
            leaf_path = "test_tree_integration.jpg"
            
        with open(leaf_path, "rb") as f:
            leaf_bytes = f.read()

        # Run pipeline
        res = await pipeline.run_pipeline(
            full_tree_bytes=tree_bytes,
            closeup_bytes=leaf_bytes,
            full_tree_filename="Mango_000000.jpg",
            closeup_filename="Mango_000000.jpg"
        )
        
        print("PIPELINE RESPONSE (JSON):")
        print(res.model_dump_json(indent=2))
        
    except Exception as e:
        print("ERROR:", e)

if __name__ == "__main__":
    asyncio.run(main())
