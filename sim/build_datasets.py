"""
Builds and saves the complete research-grade dataset split for training, validation, and testing.
"""

from pathlib import Path
from sim.generator import MissionGenerator


def main():
    data_dir = "data"
    generator = MissionGenerator(config_dir="config")
    
    # 1. Train Split: 15 missions per attack class (135 missions total)
    generator.generate_dataset_split(
        output_dir=data_dir,
        split_name="train",
        missions_per_attack=15,
        mission_duration_s=30.0
    )
    
    # 2. Validation Split: 4 missions per attack class (36 missions total)
    generator.generate_dataset_split(
        output_dir=data_dir,
        split_name="val",
        missions_per_attack=4,
        mission_duration_s=30.0
    )
    
    # 3. Test Split: 4 missions per attack class (36 missions total)
    generator.generate_dataset_split(
        output_dir=data_dir,
        split_name="test",
        missions_per_attack=4,
        mission_duration_s=30.0
    )
    
    print("\nDataset generation complete! Files saved to data/")


if __name__ == "__main__":
    main()
