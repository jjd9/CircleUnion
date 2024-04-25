# Circle Union

This repo implements two solutions to the problem:
Given N circles in a plane, determine the total area they cover.

- approx_circle_union.py - Calculates the approximate solution, but the grid resolution needs to be chosen appropriately. It is not very memory efficient.
- circle_union.py - Calculates the exact solution, but scales poorly with the number of circle intersections because of the algorithmic complexity of finding cycles in large, undirected graphs.
