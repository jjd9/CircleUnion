"""
Code to calculate the approximate area covered by N circles.
The core idea is to rasterize the circles onto a
binary image with fine enough resolution, and then count the number of pixels corresponding to the circle areas.

Author: John D'Angelo
"""


from math import pi
import cv2
import numpy as np

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def to_list(self):
        return (self.x, self.y)

class Circle:
    def __init__(self, center, r):
        self.center = center
        self.r = r
        self.r_squared = r*r
        self.area = pi * self.r_squared

    def area(self):
        return pi * self.r * self.r

def generate_circle_of_circles(num_circles, radius, radius2):
    import numpy as np

    # Calculate positions for smaller circles
    theta = np.linspace(0, 2 * np.pi, num_circles, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Plot smaller circles
    circles = []
    for i in range(num_circles):
        circles.append(Circle(Point(x[i], y[i]), radius2))
    return circles


if __name__ == "__main__":
    # center0 = Point(0,0)
    # center1 = Point(0.3, 0.4)
    # center2 = Point(0.3, 1.6)
    # center3 = Point(0.7, -0.8)
    # circle0 = Circle(center0, 1)
    # circle1 = Circle(center1, 1)
    # circle2 = Circle(center2, 0.5)
    # circle3 = Circle(center3, 0.5)

    # circles = [circle0, circle1, circle2, circle3]

    circles = generate_circle_of_circles(10, 1, 0.5)

    # choose a resolution
    res = 0.0025

    # create the grid
    max_dim = -1e6
    min_dim = 1e6
    for c in circles:
        max_dim = max(max_dim, c.center.x + c.r*2)
        max_dim = max(max_dim, c.center.x - c.r*2)
        min_dim = min(min_dim, c.center.x + c.r*2)
        min_dim = min(min_dim, c.center.x - c.r*2)
        max_dim = max(max_dim, c.center.y + c.r*2)
        max_dim = max(max_dim, c.center.y - c.r*2)
        min_dim = min(min_dim, c.center.y + c.r*2)
        min_dim = min(min_dim, c.center.y - c.r*2)

    height = width = int(round((max_dim - min_dim) / res))
    grid = np.zeros((height, width), dtype=np.uint8)


    # rasterize the circles
    for c in circles:
        grid = cv2.circle(grid, (int((c.center.x - min_dim) / res), int((c.center.y - min_dim) / res)), color=255, radius= int(c.r/res), thickness=-1)

    # calculate the area by counting
    area = np.count_nonzero(grid) * (res**2)
    print(f"Area {area}")
    cv2.imshow("test", grid)
    cv2.waitKey(0)