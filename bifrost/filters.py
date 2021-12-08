import numpy as np


class Filter:

    __slots__ = ['attribute', 'lower_bound', 'upper_bound']

    def __init__(self, attribute, lower_bound=-np.inf, upper_bound=np.inf):
        """
        A class for creating constraints on which spectra will be included in stacking.

        :param attribute: str
            The attribute of the Spectrum class which the constrain applies to.
        :param lower_bound: float
            The lower bound on the attribute.  Defaults to -inf.
        :param upper_bound: float
            The upper bound on the attribute.  Defaults to +inf.
        """
        self.attribute = attribute
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __repr__(self):
        return f"A filter with the constraint: {self.lower_bound:.2e} < {self.attribute} < {self.upper_bound:.2e}"

    def __str__(self):
        return f"{self.lower_bound:.2e} < {self.attribute} < {self.upper_bound:.2e}"

    @classmethod
    def from_str(cls, string):
        """
        Create a filter class by parsing a string with a condition in it.
        :param string: str
            The string containing the condition.  i.e. "z > 0.5"
        :return cls: Filter
            The filter class created from the string.
        """
        lb = -np.inf
        ub = np.inf
        at = None
        if '<' in string:
            elements = string.split('<')
            l = len(elements)
            if l == 3:
                lb = float(elements[0])
                at = elements[1]
                ub = float(elements[2])
            else:
                try:
                    ub = float(elements[1])
                    at = elements[0]
                except ValueError:
                    lb = float(elements[0])
                    at = elements[1]
        elif '>' in string:
            elements = string.split('>')
            l = len(elements)
            if l == 3:
                ub = float(elements[0])
                at = elements[1]
                lb = float(elements[2])
            else:
                try:
                    lb = float(elements[1])
                    at = elements[0]
                except ValueError:
                    ub = float(elements[0])
                    at = elements[1]
        at = at.strip()
        return cls(attribute=at, lower_bound=lb, upper_bound=ub)
