import math

from PIL import Image
from lib import operation as op, primitives as pr, tf_helper as tfh, matrix_builder as m
from lib.tracer import Tracer


def scenery(x, y, z):
    f = op.blend(
        op.tx(pr.sphere(1.0), m.tr(0, 0.2, 0)),
        op.tx(pr.sphere(1.0), m.tr(0, 0.7, 0))
    )
    f = op.blend(
        f,
        op.tx(pr.torus(1.3, 0.3), m.rt([0, 0, 1], math.pi / 2.0))
    )

    f = pr.cylinder(0.8)

    f = pr.zernike_surface([2.0, 1.0], 1.0)

    return f(x, y, z)


tracer = Tracer()
trace = tracer.trace(scenery)

image_data = trace.session.run(trace.image)
img = Image.fromarray(image_data, 'RGB')
img.show()
img.save("pic.png")

# jpeg_data = session.run(render)
