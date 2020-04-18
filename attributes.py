from abc import abstractmethod
from collections import Counter

import numpy


class Attribute:

    def __init__(self, name: str) -> None:
        self.name = name
        self.values = []

    def add_value(self, value) -> None:
        self.values.append(value)

    def num_of_values(self) -> int:
        return len(list(filter(lambda x: x is not None, self.values)))

    @abstractmethod
    def fill_missing_values(self) -> None:
        pass


class NumericAttribute(Attribute):

    def min_value(self) -> float:
        return min(filter(lambda x: x is not None, self.values))

    def max_value(self) -> float:
        return max(filter(lambda x: x is not None, self.values))

    def average(self) -> float:
        values = list(filter(lambda x: x is not None, self.values))

        return sum(values) / len(values)

    def fill_missing_values(self) -> None:
        avg = self.average()

        for index, value in enumerate(self.values):
            if not value:
                self.values[index] = avg

    def remove_outliers(self, m=2):
        average = self.average()
        mean_value = numpy.mean(self.values)
        deviation = numpy.std(self.values)

        for index, value in enumerate(self.values):
            if abs(value - mean_value) > m * deviation:
                self.values[index] = average


class CategoricalAttribute(Attribute):

    def mode(self, index: int) -> str:
        if index not in [1, 2]:
            raise ValueError("Invalid mode index, supported indexes: 1 and 2")

        counter = Counter(list(filter(lambda x: x is not None, self.values)))

        values = list(set(counter.values()))
        values.sort(reverse=True)

        value_index = index - 1

        if value_index >= len(values):
            value_index = len(values) - 1

        max_value = values[value_index]

        return list(counter.keys())[list(counter.values()).index(max_value)]

    def fill_missing_values(self) -> None:
        mode = self.mode(1)

        for index, value in enumerate(self.values):
            if not value:
                self.values[index] = mode
