# constants.py

"""
Stores constant mappings and dictionaries used across the project.
Separating these from the main logic improves readability.
"""

# Mapping of protein families to their member target names
FAMILIES_DICT = {
    'KDM': ['JMJD2', 'KDM4E', 'LSD1', 'KDM4C', 'KDM5B', 'KDM5A', 'KDM2B', 'KDM5C', 'KDM4B', 'KDM2A', 'PHF8', 'JMJD3', 'JMJD4', 'KDM4D', 'LSD2'],
    'HDAC': ['HDAC1', 'HDAC6', 'HDAC3', 'HDAC8', 'HDAC2', 'SIRT2', 'SIRT1', 'HDAC11', 'HDAC10', 'HDAC4', 'HDAC7', 'HDAC5', 'HDAC9', 'SIRT3', 'SIRT6', 'SIRT7'],
    'HAT': ['KAT2A', 'CREBBP', 'NCOA3', 'P300', 'NCOA1', 'PCAF', 'MYST1'],
    'PMT': ['EZH2', 'PRMT5', 'NSD2', 'PRMT4', 'DOT1L', 'PRMT6', 'SMYD3', 'PRMT1', 'PRMT3', 'SMYD2', 'PRMT8', 'SETD8', 'SETD7', 'SUV39H1', 'G9a'],
    'DNMT': ['DNMT3A', 'DNMT1', 'DNMT3B'],
    'reader': ['BAZ1A', 'BAZ2A', 'BRD8', 'PBRM1', 'BRWD1', 'SMARCA4', 'BRD9', 'BRPF1', 'TP53BP1', 'CBX7', 'BRD2', 'BRD7', 'L3MBTL3', 'L3MBTL1', 'BRD4', 'BRDT', 'BRPF3', 'WRD5', 'BAZ2B', 'BRD1', 'SMARCA2', 'BRD3']
}

# Mapping of task index to target name
IDX_TO_NAME_DICT = {
    0: 'JMJD2', 1: 'KAT2A', 2: 'KDM4E', 3: 'HDAC1', 4: 'HDAC6', 5: 'HDAC3', 6: 'HDAC8', 7: 'HDAC2',
    8: 'LSD1', 9: 'SIRT2', 10: 'KDM4C', 11: 'EZH2', 12: 'KDM5B', 13: 'KDM5A', 14: 'PRMT5', 15: 'SIRT1',
    16: 'HDAC11', 17: 'KDM2B', 18: 'HDAC10', 19: 'CREBBP', 20: 'NSD2', 21: 'HDAC4', 22: 'HDAC7',
    23: 'HDAC5', 24: 'NCOA3', 25: 'HDAC9', 26: 'PRMT4', 27: 'SIRT3', 28: 'P300', 29: 'KDM5C', 30: 'DOT1L',
    31: 'PRMT6', 32: 'KDM4B', 33: 'SMYD3', 34: 'KDM2A', 35: 'PRMT1', 36: 'PHF8', 37: 'PRMT3', 38: 'JMJD3',
    39: 'NCOA1', 40: 'PCAF', 41: 'SMYD2', 42: 'PRMT8', 43: 'JMJD4', 44: 'KDM4D', 45: 'SETD8', 46: 'SIRT6',
    47: 'SETD7', 48: 'LSD2', 49: 'SIRT7', 50: 'SUV39H1', 51: 'MYST1', 52: 'BAZ1A', 53: 'BAZ2A', 54: 'BRD8',
    55: 'PBRM1', 56: 'BRWD1', 57: 'SMARCA4', 58: 'DNMT3A', 59: 'BRD9', 60: 'BRPF1', 61: 'TP53BP1',
    62: 'CBX7', 63: 'BRD2', 64: 'BRD7', 65: 'L3MBTL3', 66: 'L3MBTL1', 67: 'BRD4', 68: 'BRDT', 69: 'BRPF3',
    70: 'DNMT1', 71: 'WRD5', 72: 'BAZ2B', 73: 'BRD1', 74: 'SMARCA2', 75: 'BRD3', 76: 'DNMT3B', 77: 'G9a'
}