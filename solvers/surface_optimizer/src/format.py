size_units = [(1, 'B'), (2 ** 10, 'KB'), (2 ** 20, 'MB'), (2 ** 30, 'GB')]
time_units = [(1, 'ns'), (1e3, 'us'), (1e6, 'ms'), (1e9, 's'),
              (60 * 1e9, 'min'), (60 * 60 * 1e9, 'hr')]


def best_unit(x, units):
    """
    Finds the index of the largest unit in `units` that is less than or equal to `x`. This allows for selecting the most appropriate scale or level of detail for a given value.
    
    Args:
        x (float): The value to compare against the units.
        units (list of tuple): A list of tuples, where each tuple contains a unit value (float) and its label (str).
    
    Returns:
        int: The index of the largest unit less than or equal to x.  Returns the last index if x is greater than or equal to all unit values.
    """
    i = 0
    while i + 1 < len(units) and x >= units[i + 1][0]:
        i += 1
    return i


def fmt(x, units):
    """
    Formats a numerical value by scaling it to the most appropriate unit.
    
    Args:
        x (float): The numerical value to format.
        units (list of tuples): A list of tuples, where each tuple contains a scaling factor and a unit string. 
                                 For example: [(1, 'm'), (1000, 'mm'), (1000000, 'µm')]
    
    Returns:
        str: A string representing the formatted value with units, rounded to two decimal places.
    """
    i = best_unit(x, units)
    return f'{x / units[i][0]:.2f} {units[i][1]}'


def sizefmt(x):
    """
    Formats a number into a human-readable string, using appropriate units for clarity.
    
    Args:
        x (float or int): The numerical value to be formatted.
    
    Returns:
        str: A string representation of the number with a suitable size unit appended.
    """
    return fmt(x, size_units)


def tfmt(t):
    """
    Transforms a time value into a formatted string representing nanoseconds.
    
    Args:
        t (float): The time value in seconds to format.
    
    Returns:
        str: The formatted time string, scaled to nanoseconds and using the `time_units` format.
    """
    return fmt(t * 1e9, time_units)
