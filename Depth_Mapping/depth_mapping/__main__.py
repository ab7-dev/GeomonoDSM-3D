"""Entry point for  ``python -m depth_mapping``  (depth inference CLI).

Usage
-----
From the project root (Depth_Mapping_Part_1/):

    python -m depth_mapping --image Input/test_2.tif --output-dir outputs/

    python -m depth_mapping --image Input/test_1.jpeg --fast

    python -m depth_mapping --image Input/sfo3.tif \\
        --checkpoint-dir D:/Collaborate_Projects/SIH/checkpoints

    # Override checkpoint directory via environment variable
    set DA_CHECKPOINT_DIR=D:\\Collaborate_Projects\\SIH\\checkpoints
    python -m depth_mapping --image Input/innsbruck3.tif
"""

from depth_mapping.depth import main

main()
