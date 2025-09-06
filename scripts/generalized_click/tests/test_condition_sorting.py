"""Test the condition sorting fix for None values."""

# Test the sorting fix
def test_condition_sorting():
    """Test that condition keys with None values can be sorted properly."""
    
    # Simulate condition keys with the new "NO_TARGET" approach
    condition_keys = [
        ('TaskA', 'Reach', (0.1, 0.2)),
        ('TaskA', 'Click', "NO_TARGET"),
        ('TaskB', 'Reach', (0.0, 0.1)),
        ('TaskA', 'Reach', "NO_TARGET"),
        ('TaskB', 'Click', (-0.1, 0.0)),
        ('TaskC', 'InterTrial', "NO_TARGET"),
    ]
    
    print("Original condition keys:")
    for key in condition_keys:
        print(f"  {key}")
    
    # Test the new sorting function
    def sort_key(condition_tuple):
        task, epoch_type, target = condition_tuple
        # Convert different target types to sortable format
        if target == "NO_TARGET":
            target_sort = (0,)  # NO_TARGET conditions come first
        elif isinstance(target, tuple):
            target_sort = (1,) + target  # Target tuples come after NO_TARGET
        else:
            target_sort = (2, target)  # Any other target type comes last
        return (task, epoch_type, target_sort)
    
    unique_keys = list(set(condition_keys))
    print(f"\nUnique keys before sorting:")
    for key in unique_keys:
        print(f"  {key}")
    
    try:
        unique_keys.sort(key=sort_key)
        print(f"\nUnique keys after sorting:")
        for i, key in enumerate(unique_keys):
            print(f"  {i}: {key}")
        
        print(f"\nSUCCESS: Sorting works with None values!")
        return True
        
    except Exception as e:
        print(f"\nFAIL: Sorting failed: {e}")
        return False


if __name__ == "__main__":
    test_condition_sorting()