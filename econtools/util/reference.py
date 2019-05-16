from typing import Dict


def state_name_to_fips(name: str) -> int:
    """Take state name and return fips as int

    Args:
        x (str): State name (e.g., Arizona).

    Returns:
        int: FIPS code.
    """
    return _name_to_fips_xwalk[name]


def state_name_to_abbr(name: str) -> str:
    """Take state name, return 2-letter state abbreviation

    Args:
        x (str): State name (e.g., Idaho)

    Returns:
        str: 2-letter state abbreviation (e.g., ID)
    """
    return _state_name_to_abbr[name]


def state_fips_to_name(fips: int) -> str:
    """Take fips as int and return state name

    Args:
        x (int): State FIPS.

    Returns:
        str: State name (e.g., Colorado)
    """
    return _fips_to_name_xwalk[fips]


def state_abbr_to_name(abbr: str) -> str:
    """Take 2-letter state abbreviation, return name

    Args:
        x (str): 2-letter state abbreviation (e.g., CA)

    Returns:
        str: State name (e.g., California)
    """
    return _state_abbr_to_name[abbr]


def state_fips_to_abbr(fips: int) -> str:
    return state_name_to_abbr(state_fips_to_name(fips))


def state_abbr_to_fips(abbr: str) -> int:
    return state_name_to_fips(state_abbr_to_name(abbr))


_fips_to_name_xwalk: Dict[int, str] = {
    1: 'Alabama',
    2: 'Alaska',
    4: 'Arizona',
    5: 'Arkansas',
    6: 'California',
    8: 'Colorado',
    9: 'Connecticut',
    10: 'Delaware',
    11: 'District of Columbia',
    12: 'Florida',
    13: 'Georgia',
    15: 'Hawaii',
    16: 'Idaho',
    17: 'Illinois',
    18: 'Indiana',
    19: 'Iowa',
    20: 'Kansas',
    21: 'Kentucky',
    22: 'Louisiana',
    23: 'Maine',
    24: 'Maryland',
    25: 'Massachusetts',
    26: 'Michigan',
    27: 'Minnesota',
    28: 'Mississippi',
    29: 'Missouri',
    30: 'Montana',
    31: 'Nebraska',
    32: 'Nevada',
    33: 'New Hampshire',
    34: 'New Jersey',
    35: 'New Mexico',
    36: 'New York',
    37: 'North Carolina',
    38: 'North Dakota',
    39: 'Ohio',
    40: 'Oklahoma',
    41: 'Oregon',
    42: 'Pennsylvania',
    44: 'Rhode Island',
    45: 'South Carolina',
    46: 'South Dakota',
    47: 'Tennessee',
    48: 'Texas',
    49: 'Utah',
    50: 'Vermont',
    51: 'Virginia',
    53: 'Washington',
    54: 'West Virginia',
    55: 'Wisconsin',
    56: 'Wyoming',
    60: 'American Samoa',
    66: 'Guam',
    69: 'Northern Mariana',
    72: 'Puerto Rico',
    78: 'Virgin Islands',
}

_state_abbr_to_name: Dict[str, str] = {
    'AK': 'Alaska',
    'AL': 'Alabama',
    'AR': 'Arkansas',
    'AS': 'American Samoa',
    'AZ': 'Arizona',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DC': 'District of Columbia',
    'DE': 'Delaware',
    'FL': 'Florida',
    'GA': 'Georgia',
    'GU': 'Guam',
    'HI': 'Hawaii',
    'IA': 'Iowa',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'MA': 'Massachusetts',
    'MD': 'Maryland',
    'ME': 'Maine',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MO': 'Missouri',
    'MP': 'Northern Mariana Islands',
    'MS': 'Mississippi',
    'MT': 'Montana',
    'NA': 'National',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'NE': 'Nebraska',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NV': 'Nevada',
    'NY': 'New York',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'PR': 'Puerto Rico',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VA': 'Virginia',
    'VI': 'Virgin Islands',
    'VT': 'Vermont',
    'WA': 'Washington',
    'WI': 'Wisconsin',
    'WV': 'West Virginia',
    'WY': 'Wyoming'
}

_name_to_fips_xwalk = {v: k
                       for k, v in _fips_to_name_xwalk.items()}

_state_name_to_abbr = {v: k
                       for k, v in _state_abbr_to_name.items()}

state_fips_list = tuple(_fips_to_name_xwalk.keys())

state_names_list = tuple(_fips_to_name_xwalk.values())

state_abbr_list = tuple(_state_abbr_to_name.keys())
