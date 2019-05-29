import numpy as np
import camb_cosmo
import fit_linP
import recons_cosmo
import thermal_model

class LikelihoodParameter(object):
    """Base class for likelihood parameter"""

    def __init__(self,name,min_value,max_value,value=None): 
        """Base class for parameter used in likelihood"""

        self.name=name
        self.min_value=min_value
        self.max_value=max_value
        self.value=value
        return


    def value_in_cube(self):
        """Normalize parameter value to [0,1]."""

        assert self.value, 'value not set in parameter '+self.name
        return (self.value-self.min_value)/(self.max_value-self.min_value)


    def set_from_cube(self,x):
        """Set parameter value from value in cube [0,1]."""

        value=self.value_from_cube(x)
        self.value=value
        return

    def info_str(self,all_info=False):
        """Return a string with parameter name and value, for debugging"""

        info=self.name+' = '+str(self.value)
        if all_info:
            info+=' , '+str(self.min_value)+' , '+str(self.max_value)

        return info


    def value_from_cube(self,x):
        """Given the value in range (xmin,xmax), return absolute value"""

        return self.min_value+x*(self.max_value-self.min_value)


    def is_same_parameter(self,param):
        """Check whether input parameter is the same parameter.
            It checks name and range, not actual value."""

        if self.name != param.name:
            return False
        if self.min_value != param.min_value:
            return False
        if self.max_value != param.max_value:
            return False

        return True
