#!/usr/bin/env python3
"""
Update existing embeddings metadata to include library information
without recalculating the embeddings.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
import argparse
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_module_descriptions(package_name: str, descriptions_dir: Path) -> Optional[Dict]:
    """Load module descriptions for a package to get library information."""
    desc_file = descriptions_dir / f"{package_name}.json"
    if not desc_file.exists():
        logger.warning(f"Module descriptions not found for {package_name}")
        return None
    
    try:
        with open(desc_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load descriptions for {package_name}: {e}")
        return None


def update_package_metadata(package_name: str, embeddings_dir: Path, descriptions_dir: Path) -> bool:
    """Update metadata for a single package to include library information."""
    # Load existing metadata
    metadata_path = embeddings_dir / "packages" / package_name / "metadata.json"
    if not metadata_path.exists():
        logger.warning(f"Metadata not found for {package_name}")
        return False
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load metadata for {package_name}: {e}")
        return False
    
    # Load module descriptions to get library info
    descriptions = load_module_descriptions(package_name, descriptions_dir)
    if not descriptions:
        return False
    
    # Build module to library mapping
    module_to_library = {}
    libraries = descriptions.get("libraries", {})
    for library_name, library_data in libraries.items():
        modules = library_data.get("modules", {})
        for module_name in modules:
            module_to_library[module_name] = library_name
    
    # Update metadata with library information
    modules_updated = 0
    for module_info in metadata.get("modules", []):
        module_path = module_info.get("module_path")
        if module_path and module_path in module_to_library:
            module_info["library"] = module_to_library[module_path]
            modules_updated += 1
    
    # Save updated metadata
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Updated {package_name}: {modules_updated}/{len(metadata.get('modules', []))} modules with library info")
        return True
    except Exception as e:
        logger.error(f"Failed to save metadata for {package_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Update embeddings metadata with library information")
    parser.add_argument("--embeddings-dir", default="package-embeddings",
                        help="Directory containing package embeddings")
    parser.add_argument("--descriptions-dir", default="module-descriptions",
                        help="Directory containing module descriptions")
    parser.add_argument("--packages", nargs="+",
                        help="Specific packages to update (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be updated without making changes")
    
    args = parser.parse_args()
    
    embeddings_dir = Path(args.embeddings_dir)
    descriptions_dir = Path(args.descriptions_dir)
    
    if not embeddings_dir.exists():
        logger.error(f"Embeddings directory not found: {embeddings_dir}")
        return
    
    if not descriptions_dir.exists():
        logger.error(f"Descriptions directory not found: {descriptions_dir}")
        return
    
    # Get list of packages to process
    if args.packages:
        packages = args.packages
    else:
        # Get all packages with embeddings
        packages_dir = embeddings_dir / "packages"
        if not packages_dir.exists():
            logger.error(f"Packages directory not found: {packages_dir}")
            return
        packages = [p.name for p in packages_dir.iterdir() if p.is_dir()]
    
    logger.info(f"Found {len(packages)} packages to update")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
        for package in packages[:10]:  # Show first 10 as example
            desc_file = descriptions_dir / f"{package}.json"
            if desc_file.exists():
                logger.info(f"Would update: {package}")
        logger.info(f"... and {len(packages) - 10} more packages")
        return
    
    # Process each package
    success_count = 0
    for package in tqdm(packages, desc="Updating metadata"):
        if update_package_metadata(package, embeddings_dir, descriptions_dir):
            success_count += 1
    
    logger.info(f"Successfully updated {success_count}/{len(packages)} packages")


if __name__ == "__main__":
    main()