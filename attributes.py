from abc import abstractmethod
from collections import Counter
from math import floor
from math import ceil
from math import sqrt
from math import pow
import numpy


class Attribute:

    def __init__(self, name: str) -> None:
        self.name = name
        self.values = []

    def add_value(self, value) -> None:
        self.values.append(value)

    def num_of_values(self) -> int:
        return len(list(filter(lambda x: x is not None, self.values)))

    def percent_of_missing_values(self) -> float:
        empty_values = 0

        for value in self.values:
            if value:
                continue

            empty_values += 1

        return empty_values / len(self.values) * 100

    def cardinality(self) -> int:
        return len(Counter(self.values).keys())

    @abstractmethod
    def fill_missing_values(self) -> None:
        pass


class NumericAttribute(Attribute):

    def min_value(self) -> float:
        return min(filter(lambda x: x is not None, self.values))

    def max_value(self) -> float:
        return max(filter(lambda x: x is not None, self.values))

    def quartile(self, index: int) -> float:
        if index not in [1, 2, 3]:
            raise ValueError("Invalid quartile index, possible indexes: 1, 2, 3")

        values = list(filter(lambda x: x is not None, self.values))

        values.sort()

        index = len(values) * 0.25 * index

        if index % 1 != 0:
            return (values[floor(index)] + values[ceil(index)]) / 2
        else:
            return values[int(index)]

    def average(self) -> float:
        values = list(filter(lambda x: x is not None, self.values))

        return sum(values) / len(values)

    def standard_deviation(self) -> float:
        values = list(filter(lambda x: x is not None, self.values))

        values = list(map(lambda x: x * x, values))

        return sqrt(sum(values) / len(values) - pow(self.average(), 2))

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

    def get_normalized_values(self) -> list:
        new_values = []
        minimum = self.min_value()
        maximum = self.max_value()

        for value in self.values:
            new_values.append(((value - minimum) / (maximum - minimum)) * (1))

        return new_values


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


def print_numeric_attribute_info(attributes: [Attribute]) -> None:
    print('{:>20} {:>20} {:>20} {:>20} {:>20} {:>20} {:>20} {:>20} {:>20} {:>20} {:>20}'.format("Pavadinimas", "kiekis",
                                                                                                "missing value %",
                                                                                                "kardinalumas", "min",
                                                                                                "max", "1 quart",
                                                                                                "3 quart", "average",
                                                                                                "mediana", "deviation"))

    for attribute in attributes:
        if not isinstance(attribute, NumericAttribute):
            continue

        print('{:>20} {:>20} {:>20} {:>20} {:>20} {:>20} {:>20} {:>20} {:>20} {:>20} {:>20}'.format(
            attribute.name,
            attribute.num_of_values(),
            attribute.percent_of_missing_values(),
            attribute.cardinality(),
            attribute.min_value(),
            attribute.max_value(),
            attribute.quartile(1),
            attribute.quartile(3),
            attribute.average(),
            attribute.quartile(2),
            attribute.standard_deviation()
        ))
