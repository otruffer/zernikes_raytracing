import math

from PIL import Image
from lib import operation as op, primitives as pr, tf_helper as tfh, matrix_builder as m
from lib.tracer import Tracer, TracerOptions


def scenery(x, y, z):
    f = op.blend(
        op.tx(pr.sphere(1.0), m.tr(0, 0.2, 0)),
        op.tx(pr.sphere(1.0), m.tr(0, 0.7, 0))
    )
    f = op.blend(
        f,
        op.tx(pr.torus(1.3, 0.3), m.rt([0, 0, 1], math.pi / 2.0))
    )

    # f = pr.cylinder(0.8)

    f = pr.zernike_surface([0.0, 0.000, 0.0, 0.0, -0.0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.05], 1.0)

    return f(x, y, z)


options = TracerOptions()
options.resolution = (600, 600)
options.camera_position = [0.0, 0.0, 10.0]
options.v_unit = [-1.0, 0.0, 0.0]
options.lookAt = (0.0, 0.0, 0.0)
options.verbose = True
options.speed_up = 1.0
options.damping = 1.0
options.max_steps = 1000
options.focal_length = 10.0

tracer = Tracer(options)
trace = tracer.trace(scenery)

image_data = trace.session.run(trace.image)
img = Image.fromarray(image_data, 'RGB')
img.show()
img.save("pic.png")

# jpeg_data = session.run(render)
