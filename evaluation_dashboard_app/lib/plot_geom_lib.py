import os
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Polygon, Point
from matplotlib.patches import Polygon as PlotPolygon
from typing import List, Tuple


def calculate_common_area(rect_list1: List[List[Tuple]], rect_list2: List[List[Tuple]]) -> float:
    """
    Calculate the total common area between two lists of rectangles.

    Args:
        rect_list1: First list of rectangles, each containing 4 (x,y) coordinates
        rect_list2: Second list of rectangles, each containing 4 (x,y) coordinates

    Returns:
        float: Total common area
    """
    try:
        # Convert all rectangles to Polygons
        polys1 = [Polygon(rect) for rect in rect_list1]
        polys2 = [Polygon(rect) for rect in rect_list2]

        # Combine all polygons in each list using union
        if len(polys1) == 0:
            return 0.0
        combined1 = polys1[0]
        for poly in polys1[1:]:
            combined1 = combined1.union(poly)

        if len(polys2) == 0:
            return 0.0
        else:
            combined2 = polys2[0]
            for poly in polys2[1:]:
                combined2 = combined2.union(poly)

        # Calculate intersection area
        intersection = combined1.intersection(combined2)
        return intersection.area / combined1.union(combined2).area

    except Exception as e:
        print(f"Error calculating common area: {e}")
        return 0.0


def calc_line_polygon_distance(
    line_points: List[Tuple[float, float]], polygon_points: List[Tuple[float, float]]
) -> float:
    """
    Calculate minimum distance between a 2D line and a polygon.

    Args:
        line_points: List of two points [(x1,y1), (x2,y2)] defining the line
        polygon_points: List of points [(x1,y1), (x2,y2), ...] defining the polygon

    Returns:
        float: Minimum distance between line and polygon.
               Returns 0.0 if they intersect.
    """
    try:
        # Create Shapely objects
        line = LineString(line_points)
        polygon = Polygon(polygon_points)

        # If they intersect, distance is 0
        if line.intersects(polygon):
            return 0.0

        # Calculate minimum distance
        return line.distance(polygon)

    except Exception as e:
        print(f"Error calculating distance: {e}")
        return float("inf")


def calc_line_vector_intersection_time(
    line_points: List[Tuple[float, float]],
    vector_start: Tuple[float, float],
    vector_direction: Tuple[float, float],
    vehicle_polygon_points: List[Tuple[float, float]],
) -> Tuple[float, float, float]:
    """
    Calculate intersection distance, time, and start point distance to vehicle polygon.

    Args:
        line_points: List of two points [(x1,y1), (x2,y2)] defining the line segment
        vector_start: Starting point (x,y) of the vector
        vector_direction: Direction vector (dx,dy) representing velocity
        vehicle_polygon_points: List of points [(x1,y1), (x2,y2), ...] defining vehicle polygon

    Returns:
        Tuple[float, float, float]:
            - Distance from intersection to vehicle polygon boundary (10.0 if no intersection)
            - Time to intersection (distance/speed) (10.0 if no intersection)
            - Distance from vector start point to vehicle polygon boundary
    """
    try:
        # Extract points
        p1 = np.array(line_points[0])
        p2 = np.array(line_points[1])
        p3 = np.array(vector_start)
        p4 = np.array(vector_start) + np.array(vector_direction)

        # Create vehicle polygon and start point
        vehicle_polygon = Polygon(vehicle_polygon_points)
        start_point = Point(vector_start)

        # Calculate start point distance to vehicle polygon
        distance = start_point.distance(vehicle_polygon.boundary)

        # Calculate vector speed (magnitude of direction vector)
        vector_speed = np.linalg.norm(vector_direction)
        if vector_speed < 1e-10:  # Check for zero velocity
            return (
                1000.0,
                distance,
                10.0,
            )

        # Calculate denominator
        denominator = ((p4[1] - p3[1]) * (p2[0] - p1[0])) - ((p4[0] - p3[0]) * (p2[1] - p1[1]))

        # Check if lines are parallel
        if abs(denominator) < 1e-10:
            return (
                1000.0,
                distance,
                10.0,
            )

        # Calculate ua (intersection parameter for vector)
        ua = (
            ((p4[0] - p3[0]) * (p1[1] - p3[1])) - ((p4[1] - p3[1]) * (p1[0] - p3[0]))
        ) / denominator

        # Calculate ub (intersection parameter for line segment)
        ub = (
            ((p2[0] - p1[0]) * (p1[1] - p3[1])) - ((p2[1] - p1[1]) * (p1[0] - p3[0]))
        ) / denominator

        # Check if intersection occurs within line segment and in vector direction
        if 0.0 <= ua and ub >= 0:
            # Calculate intersection point
            intersection_x = p1[0] + (ua * (p2[0] - p1[0]))
            intersection_y = p1[1] + (ua * (p2[1] - p1[1]))
            intersection_point = Point(intersection_x, intersection_y)

            # Calculate distance to polygon boundary first
            intersection_distance = intersection_point.distance(vehicle_polygon.boundary)

            # Then calculate time (distance/speed)
            time = intersection_point.distance(start_point) / vector_speed
            # print("intersection_point", intersection_point, "vector_speed", vector_speed, "time", time)

            return intersection_distance, distance, time

        return (
            1000.0,
            distance,
            10.0,
        )

    except Exception as e:
        print(f"Error calculating intersection: {e}")
        return 1000.0, 1000.0, 10.0


def plot_rectangles_with_common_area(
    rect_list1: List[List[Tuple]],
    rect_list2: List[List[Tuple]],
    result_directory: str,
    index: int,
    output_folder: str,
    title: str = "Rectangle Comparison",
    vehicle_polygon: List[Tuple] = None,
    area_polygon: List[Tuple] = None,
):
    """
    Plot rectangles with vehicle and area polygons labeled directly on plot.
    """
    try:
        plt.figure(figsize=(10, 10))
        ax = plt.gca()

        # Original rectangle plotting code...
        polys1 = [Polygon(rect) for rect in rect_list1]
        polys2 = [Polygon(rect) for rect in rect_list2]

        combined1 = polys1[0]
        for poly in polys1[1:]:
            combined1 = combined1.union(poly)

        if len(polys2) > 0:
            combined2 = polys2[0]
            if len(polys2) > 1:
                for poly in polys2[1:]:
                    combined2 = combined2.union(poly)
            intersection = combined1.intersection(combined2)
        else:
            combined2 = None
            intersection = None

        # Plot rectangles with legend
        for rect in rect_list1:
            poly = PlotPolygon(
                rect, facecolor="blue", alpha=0.3, edgecolor="blue", label="Ground Truth"
            )
            ax.add_patch(poly)

        for rect in rect_list2:
            poly = PlotPolygon(
                rect, facecolor="green", alpha=0.3, edgecolor="green", label="Estimation"
            )
            ax.add_patch(poly)

        # Plot vehicle polygon with label on plot
        if vehicle_polygon is not None:
            vehicle = PlotPolygon(
                vehicle_polygon,
                facecolor="red",
                alpha=0.3,
                edgecolor="red",  # Removed label from legend
            )
            ax.add_patch(vehicle)

            # Add label at centroid of vehicle polygon
            centroid_x = sum(x for x, _ in vehicle_polygon) / len(vehicle_polygon)
            centroid_y = sum(y for _, y in vehicle_polygon) / len(vehicle_polygon)
            plt.text(
                centroid_x,
                centroid_y,
                "Vehicle",
                horizontalalignment="center",
                verticalalignment="center",
                color="red",
                fontweight="bold",
            )

        # Plot area polygon with label on plot
        if area_polygon is not None:
            area = PlotPolygon(
                area_polygon,
                facecolor="black",
                alpha=0.3,
                edgecolor="black",  # Removed label from legend
            )
            ax.add_patch(area)

            # Add label at centroid of area polygon
            centroid_x = sum(x for x, _ in area_polygon) / len(area_polygon)
            centroid_y = sum(y for _, y in area_polygon) / len(area_polygon)
            plt.text(
                centroid_x,
                centroid_y,
                "Area0",
                horizontalalignment="center",
                verticalalignment="center",
                color="black",
                fontweight="bold",
            )

        # Plot intersection area
        if intersection is not None and not intersection.is_empty:
            if isinstance(intersection, Polygon):
                intersect_coords = list(intersection.exterior.coords)[:-1]
                poly = PlotPolygon(
                    intersect_coords,
                    facecolor="orange",
                    alpha=0.5,
                    edgecolor="orange",
                    label="Common Area",
                )
                ax.add_patch(poly)
            else:  # MultiPolygon
                for geom in intersection.geoms:
                    intersect_coords = list(geom.exterior.coords)[:-1]
                    poly = PlotPolygon(
                        intersect_coords,
                        facecolor="orange",
                        alpha=0.5,
                        edgecolor="orange",
                        label="Common Area",
                    )
                    ax.add_patch(poly)

        # Rest of the plotting code...
        plt.xlim(-20, 20)
        plt.ylim(-20, 20)
        plt.grid(True)
        plt.title(title)

        # Calculate areas
        area1 = combined1.area
        if combined2 is not None:
            area2 = combined2.area
        else:
            area2 = 0.0
        if intersection is not None and not intersection.is_empty:
            intersection_area = intersection.area
        else:
            intersection_area = 0.0

        # Add area information
        info_text = (
            f"GT Area List: {area1:.2f}\n"
            f"Est Area List: {area2:.2f}\n"
            f"Common Area: {intersection_area:.2f}\n"
            f"IoU: {intersection_area/(area1 + area2 - intersection_area):.3f}"
        )

        plt.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            fontsize=10,
        )

        # Legend only for rectangles and common area
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        if not os.path.exists(f"{result_directory}/{output_folder}"):
            os.makedirs(f"{result_directory}/{output_folder}")
        plt.savefig(
            f"{result_directory}/{output_folder}/{output_folder}_comparison{int(index):04d}.jpg",
            dpi=300,
        )
        plt.close()

    except Exception as e:
        print(f"Error plotting rectangles: {e}")
