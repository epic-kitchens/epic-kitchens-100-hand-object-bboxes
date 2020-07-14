from epic_kitchens.hoa.types import BBox, FloatVector


class TestFloatVector:
    def test_adding(self):
        v1 = FloatVector(1, 2)
        v2 = FloatVector(3, 5)
        v3 = v1 + v2

        assert v3.x == 4
        assert v3.y == 7

    def test_multiplying(self):
        v = FloatVector(1, 2)

        v2 = v * 2

        assert v2.x == 2
        assert v2.y == 4

    def test_coord(self):
        v = FloatVector(1, 2)
        x, y = v.coord
        assert x == 1
        assert y == 2

    def test_scale(self):
        v = FloatVector(1, 2)
        v.scale(width_factor=2, height_factor=4)
        assert v.x == 2
        assert v.y == 8


class TestBBox:
    def test_center(self):
        bbox = BBox(1, 3, 2, 4)
        x, y = bbox.center
        assert x == 1.5
        assert y == 3.5

    def test_center_int(self):
        bbox = BBox(1, 3, 2, 4)
        x, y = bbox.center_int
        # Python rounding is done towards the even no.
        assert x == 2
        assert y == 4

    def test_center_scale(self):
        bbox = BBox(10, 100, 30, 120)
        bbox.center_scale(2, 3)
        assert bbox.left == 0
        assert bbox.right == 40
        assert bbox.top == 80
        assert bbox.bottom == 140

    def test_coords(self):
        bbox = BBox(1, 2, 3, 4)
        ((x1, y1), (x2, y2)) = bbox.coords

        assert x1 == 1
        assert y1 == 2
        assert x2 == 3
        assert y2 == 4

    def test_coords_int(self):
        bbox = BBox(1.1, 2.1, 3.1, 4.1)
        ((x1, y1), (x2, y2)) = bbox.coords_int

        assert x1 == 1
        assert y1 == 2
        assert x2 == 3
        assert y2 == 4

    def test_width_and_height(self):
        bbox = BBox(1, 2, 3, 5)
        assert bbox.width == 2
        assert bbox.height == 3

    def test_top_left(self):
        bbox = BBox(1, 2, 3, 5)
        assert bbox.top_left == (1, 2)

    def test_bottom_right(self):
        bbox = BBox(1, 2, 3, 5)
        assert bbox.bottom_right == (3, 5)
