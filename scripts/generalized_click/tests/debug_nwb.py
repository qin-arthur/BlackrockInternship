"""Debug script to examine NWB file structure."""

from pynwb import NWBHDF5IO
from pathlib import Path
import numpy as np

def examine_nwb_file(nwb_path):
    """Examine the structure of an NWB file."""
    print(f"\nExamining NWB file: {nwb_path}")
    print("=" * 80)
    
    with NWBHDF5IO(str(nwb_path), 'r') as io:
        nwbfile = io.read()
        
        # Basic info
        print(f"\nSession ID: {nwbfile.session_id}")
        print(f"Session Description: {nwbfile.session_description}")
        print(f"Experiment Description: {nwbfile.experiment_description}")
        
        # Epochs
        print(f"\n--- EPOCHS ---")
        epochs_df = nwbfile.epochs.to_dataframe()
        print(f"Number of epochs: {len(epochs_df)}")
        print(f"Epoch columns: {list(epochs_df.columns)}")
        print(f"\nFirst 5 epochs:")
        print(epochs_df.head())
        
        if 'epoch_type' in epochs_df.columns:
            print(f"\nUnique epoch types: {epochs_df['epoch_type'].unique()}")
        if 'task' in epochs_df.columns:
            print(f"Unique tasks: {epochs_df['task'].unique()}")
        
        # Acquisition time series
        print(f"\n--- ACQUISITION TIME SERIES ---")
        for name, obj in nwbfile.acquisition.items():
            print(f"\nName: {name}")
            print(f"  Type: {type(obj).__name__}")
            if hasattr(obj, 'data'):
                print(f"  Data shape: {obj.data.shape}")
            if hasattr(obj, 'rate'):
                print(f"  Sampling rate: {obj.rate} Hz")
            if hasattr(obj, 'description'):
                print(f"  Description: {obj.description}")
            if hasattr(obj, 'timestamps') and obj.timestamps is not None:
                print(f"  Timestamps shape: {obj.timestamps.shape}")
        
        # Processing modules
        print(f"\n--- PROCESSING MODULES ---")
        for module_name, module in nwbfile.processing.items():
            print(f"\nModule: {module_name}")
            for container_name, container in module.containers.items():
                print(f"  Container: {container_name} ({type(container).__name__})")
        
        # Units (if present)
        if hasattr(nwbfile, 'units') and nwbfile.units is not None:
            print(f"\n--- UNITS ---")
            units_df = nwbfile.units.to_dataframe()
            print(f"Number of units: {len(units_df)}")
            print(f"Units columns: {list(units_df.columns)}")
        
        # Let's check the first epoch in detail
        print(f"\n--- DETAILED FIRST EPOCH ---")
        first_epoch = epochs_df.iloc[0]
        print(f"Start time: {first_epoch['start_time']}")
        print(f"Stop time: {first_epoch['stop_time']}")
        print(f"Duration: {first_epoch['stop_time'] - first_epoch['start_time']}")
        if 'task' in first_epoch:
            print(f"Task: {first_epoch['task']}")
        if 'epoch_type' in first_epoch:
            print(f"Epoch type: {first_epoch['epoch_type']}")
        
        # Check processing modules in detail
        print(f"\n--- PROCESSING MODULE DATA ---")
        for module_name, module in nwbfile.processing.items():
            for container_name, container in module.containers.items():
                print(f"\n{module_name}/{container_name}:")
                if hasattr(container, 'data'):
                    print(f"  Data shape: {container.data.shape}")
                if hasattr(container, 'rate'):
                    print(f"  Sampling rate: {container.rate} Hz")
                if hasattr(container, 'timestamps') and container.timestamps is not None:
                    print(f"  Timestamps shape: {container.timestamps.shape}")
                    print(f"  Time range: {container.timestamps[0]:.2f} - {container.timestamps[-1]:.2f} seconds")
        
        # Look for any time series that might contain spike or behavioral data
        print(f"\n--- SEARCHING FOR SPIKE/BEHAVIOR DATA PATTERNS ---")
        for name, obj in nwbfile.acquisition.items():
            lower_name = name.lower()
            if any(pattern in lower_name for pattern in ['spike', 'neural', 'unit', 'binned', 'mouse', 'position', 'velocity', 'behavior', 'target']):
                print(f"\nPotential data source: {name}")
                if hasattr(obj, 'data'):
                    print(f"  Shape: {obj.data.shape}")
                    # Check if timestamps align with epochs
                    if hasattr(obj, 'timestamps') and obj.timestamps is not None:
                        print(f"  Time range: {obj.timestamps[0]:.2f} - {obj.timestamps[-1]:.2f} seconds")
                    elif hasattr(obj, 'starting_time') and hasattr(obj, 'rate'):
                        duration = obj.data.shape[0] / obj.rate
                        print(f"  Time range: {obj.starting_time:.2f} - {obj.starting_time + duration:.2f} seconds")


def main():
    # Default path from main.py
    nwb_dir = Path("C:/Users/aqin/Downloads/GeneralizedClick/mint_pipeline_output/nwb_files")
    
    if nwb_dir.exists():
        nwb_files = list(nwb_dir.glob("*.nwb"))
        if nwb_files:
            print(f"Found {len(nwb_files)} NWB files")
            # Examine first file in detail
            examine_nwb_file(nwb_files[0])
        else:
            print(f"No NWB files found in {nwb_dir}")
    else:
        print(f"Directory does not exist: {nwb_dir}")


if __name__ == "__main__":
    main()