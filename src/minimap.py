from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import cv2
import numpy as np


DEFAULT_COURT_SIZE_M = (28.0, 15.0)
DEFAULT_TEMPLATE_SIZE = (940, 500)


def court_to_pixel(
    points_m: Iterable[Iterable[float]] | np.ndarray,
    template_size: tuple[int, int] = DEFAULT_TEMPLATE_SIZE,
    court_size_m: tuple[float, float] = DEFAULT_COURT_SIZE_M,
    margin: int = 12,
) -> np.ndarray:
    """Convert standard court metre coordinates to mini-map pixel coordinates."""
    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, 2)

    width, height = template_size
    court_length, court_width = court_size_m
    usable_width = width - 2 * margin
    usable_height = height - 2 * margin

    x = margin + points[:, 0] / court_length * usable_width
    y = margin + points[:, 1] / court_width * usable_height
    return np.column_stack([x, y])


def standard_landmarks_m() -> dict[str, tuple[float, float]]:
    """Return useful full-court reference landmarks in metres."""
    length, width = DEFAULT_COURT_SIZE_M
    key_width = 4.9
    free_throw_x = 5.8
    key_y1 = (width - key_width) / 2
    key_y2 = (width + key_width) / 2
    return {
        "left_baseline_top": (0.0, 0.0),
        "left_baseline_bottom": (0.0, width),
        "right_baseline_top": (length, 0.0),
        "right_baseline_bottom": (length, width),
        "center_top_sideline": (length / 2, 0.0),
        "center_bottom_sideline": (length / 2, width),
        "center_court": (length / 2, width / 2),
        "left_free_throw_top": (free_throw_x, key_y1),
        "left_free_throw_bottom": (free_throw_x, key_y2),
        "right_free_throw_top": (length - free_throw_x, key_y1),
        "right_free_throw_bottom": (length - free_throw_x, key_y2),
        "left_key_baseline_top": (0.0, key_y1),
        "left_key_baseline_bottom": (0.0, key_y2),
        "right_key_baseline_top": (length, key_y1),
        "right_key_baseline_bottom": (length, key_y2),
    }


def _pt(point_m: tuple[float, float], template_size: tuple[int, int], margin: int) -> tuple[int, int]:
    pixel = court_to_pixel([point_m], template_size=template_size, margin=margin)[0]
    return int(round(pixel[0])), int(round(pixel[1]))


def draw_court_template(
    template_size: tuple[int, int] = DEFAULT_TEMPLATE_SIZE,
    margin: int = 12,
) -> np.ndarray:
    """Draw a clean horizontal full-court mini-map template."""
    width, height = template_size
    court = np.full((height, width, 3), (178, 128, 78), dtype=np.uint8)
    line = (245, 245, 245)
    paint = (205, 118, 72)
    dark = (60, 60, 60)

    length_m, width_m = DEFAULT_COURT_SIZE_M
    sx = (width - 2 * margin) / length_m
    sy = (height - 2 * margin) / width_m

    top_left = (margin, margin)
    bottom_right = (width - margin, height - margin)
    cv2.rectangle(court, top_left, bottom_right, line, 2)

    center_top = _pt((length_m / 2, 0.0), template_size, margin)
    center_bottom = _pt((length_m / 2, width_m), template_size, margin)
    cv2.line(court, center_top, center_bottom, line, 2)
    cv2.circle(court, _pt((length_m / 2, width_m / 2), template_size, margin), int(1.8 * sx), line, 2)

    key_width = 4.9
    key_y1 = (width_m - key_width) / 2
    key_y2 = (width_m + key_width) / 2
    ft_x = 5.8
    hoop_x = 1.575
    hoop_y = width_m / 2

    left_key_tl = _pt((0.0, key_y1), template_size, margin)
    left_key_br = _pt((ft_x, key_y2), template_size, margin)
    right_key_tl = _pt((length_m - ft_x, key_y1), template_size, margin)
    right_key_br = _pt((length_m, key_y2), template_size, margin)

    cv2.rectangle(court, left_key_tl, left_key_br, paint, -1)
    cv2.rectangle(court, left_key_tl, left_key_br, line, 2)
    cv2.rectangle(court, right_key_tl, right_key_br, paint, -1)
    cv2.rectangle(court, right_key_tl, right_key_br, line, 2)

    cv2.circle(court, _pt((ft_x, hoop_y), template_size, margin), int(1.8 * sx), line, 2)
    cv2.circle(court, _pt((length_m - ft_x, hoop_y), template_size, margin), int(1.8 * sx), line, 2)

    left_hoop = _pt((hoop_x, hoop_y), template_size, margin)
    right_hoop = _pt((length_m - hoop_x, hoop_y), template_size, margin)
    cv2.circle(court, left_hoop, 5, dark, 2)
    cv2.circle(court, right_hoop, 5, dark, 2)

    board_half = int(0.9 * sy)
    cv2.line(court, (left_hoop[0] - 8, left_hoop[1] - board_half), (left_hoop[0] - 8, left_hoop[1] + board_half), dark, 2)
    cv2.line(court, (right_hoop[0] + 8, right_hoop[1] - board_half), (right_hoop[0] + 8, right_hoop[1] + board_half), dark, 2)

    three_r_x = int(6.75 * sx)
    three_r_y = int(6.75 * sy)
    cv2.ellipse(court, left_hoop, (three_r_x, three_r_y), 0, -68, 68, line, 2)
    cv2.ellipse(court, right_hoop, (three_r_x, three_r_y), 0, 112, 248, line, 2)

    return court


def draw_minimap_state(
    template: np.ndarray,
    players: list[dict[str, Any]],
    referees: list[dict[str, Any]],
    player_tracks: dict[int, list[tuple[float, float]]],
    referee_tracks: dict[int, list[tuple[float, float]]],
    ball: tuple[float, float] | None = None,
    ball_alpha: float = 1.0,
    colors_bgr: dict[str, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Draw players, referees, their short tracks, and the ball on a court template."""
    colors = colors_bgr or {
        "player": (255, 90, 20),
        "ball": (40, 40, 255),
        "referee": (0, 190, 255),
    }
    trail_color = (0, 215, 255)
    canvas = template.copy()
    trail_layer = canvas.copy()

    def draw_tracks(tracks: dict[int, list[tuple[float, float]]]) -> None:
        for points in tracks.values():
            if len(points) < 2:
                continue
            int_points = [(int(round(x)), int(round(y))) for x, y in points]
            for start, end in zip(int_points[:-1], int_points[1:]):
                cv2.line(trail_layer, start, end, trail_color, 3, cv2.LINE_AA)
            if int_points:
                cv2.circle(trail_layer, int_points[-1], 3, trail_color, -1, cv2.LINE_AA)

    draw_tracks(player_tracks)
    draw_tracks(referee_tracks)

    cv2.addWeighted(trail_layer, 0.35, canvas, 0.65, 0, canvas)

    def draw_people(
        people: list[dict[str, Any]],
        color: tuple[int, int, int],
        prefix: str,
    ) -> None:
        for person in people:
            point = person["point"]
            display_id = int(person["display_id"])
            centre = (int(round(point[0])), int(round(point[1])))
            cv2.circle(canvas, centre, 18, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, centre, 19, (0, 0, 0), 2, cv2.LINE_AA)
            label = f"{prefix}{display_id}"
            font_scale = 0.9
            thickness = 3
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_x = centre[0] + 24
            text_y = centre[1] - 24
            text_x = min(max(2, text_x), canvas.shape[1] - text_size[0] - 4)
            text_y = min(max(text_size[1] + 4, text_y), canvas.shape[0] - baseline - 4)
            cv2.putText(canvas, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    draw_people(players, colors["player"], "P")
    draw_people(referees, colors["referee"], "R")

    if ball is not None:
        ball_layer = canvas.copy()
        ball_centre = (int(round(ball[0])), int(round(ball[1])))
        cv2.circle(ball_layer, ball_centre, 7, colors["ball"], -1, cv2.LINE_AA)
        cv2.circle(ball_layer, ball_centre, 8, (255, 255, 255), 1, cv2.LINE_AA)
        alpha = max(0.1, min(1.0, float(ball_alpha)))
        cv2.addWeighted(ball_layer, alpha, canvas, 1.0 - alpha, 0, canvas)

    return canvas


def point_inside_template(point: tuple[float, float], template: np.ndarray) -> bool:
    """Return whether a projected point is inside the mini-map image."""
    height, width = template.shape[:2]
    x, y = point
    return 0 <= x < width and 0 <= y < height
