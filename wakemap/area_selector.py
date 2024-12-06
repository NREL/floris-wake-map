from wakemap import WakeMap

class AreaSelector():
    """
    Class for identifying candidate regions for new development based on a WakeMap.
    """
    def __init__(self, wakemap: WakeMap):
        """
        Constructor for the AreaSelector class.

        Receives instantiated WakeMap object and checks that the main maps have been 
        computed.
        """
        self.wakemap = wakemap