import sys

import tensorflow as tf

import tf_helper as tfh
import numpy as np


class Tracer:
    def __init__(self, options=None):
        self.options = options if options is not None else TracerOptions()
        self.session = None
        self.scenery = None

    def trace(self, scenery):
        resolution = self.options.resolution
        aspect_ratio = resolution[0] / resolution[1]
        min_bounds, max_bounds = (-aspect_ratio, -1), (aspect_ratio, 1)
        resolutions = list(map(lambda x: x * 1j, resolution))
        image_plane_coords = np.mgrid[min_bounds[0]:max_bounds[0]:resolutions[0],
                             min_bounds[1]:max_bounds[1]:resolutions[1]]

        # Find the center of the image plane
        camera_position = tf.constant(self.options.camera_position)
        lookAt = self.options.lookAt
        camera = camera_position - np.array(lookAt)
        camera_direction = tfh.normalize_vector(camera)
        focal_length = self.options.focal_length
        eye = camera + focal_length * camera_direction

        # Coerce into correct shape
        image_plane_center = tfh.vector_fill(resolution, camera_position)

        # Convert u,v parameters to x,y,z coordinates for the image plane
        v_unit = self.options.v_unit
        u_unit = tf.cross(camera_direction, v_unit)
        image_plane = image_plane_center + image_plane_coords[0] * tfh.vector_fill(resolution, u_unit) + \
                      image_plane_coords[1] * tfh.vector_fill(resolution, v_unit)

        # Populate the image plane with initial unit ray vectors
        initial_vectors = image_plane - tfh.vector_fill(resolution, eye)
        ray_vectors = tfh.normalize_vector(initial_vectors)
        # ray_vectors = tfh.vector_fill(resolution, -camera_direction)
        evaluated_function_nminus = tf.Variable(tf.zeros(resolution))
        damping = tf.Variable(tf.ones(resolution))
        i = tf.Variable(1.0)

        t = tf.Variable(tf.zeros_initializer(dtype=tf.float32)(resolution), name="ScalingFactor")
        space = (ray_vectors * t) + image_plane

        # Name TF ops for better graph visualization
        x = tf.squeeze(tf.slice(space, [0, 0, 0], [1, -1, -1]), axis=[0], name="X-Coordinates")
        y = tf.squeeze(tf.slice(space, [1, 0, 0], [1, -1, -1]), axis=[0], name="Y-Coordinates")
        z = tf.squeeze(tf.slice(space, [2, 0, 0], [1, -1, -1]), axis=[0], name="Z-Coordinates")

        evaluated_function = scenery(x, y, z)
        # Iteration operation
        epsilon = self.options.epsilon
        delta = 1000.0
        distance = tf.abs(evaluated_function)
        distance_no_nan = tf.where(tf.is_nan(distance), tf.ones_like(distance) * delta, distance)
        converged_bitmask = tf.less_equal(distance, epsilon)
        converged_diverged_bitmask = tf.logical_or(converged_bitmask, tf.greater(distance, delta))
        distance_step = t + (tf.sign(evaluated_function) * tf.maximum(distance * (damping), 1.0 * epsilon) * tf.to_float(
            tf.logical_not(converged_diverged_bitmask)))

        converged_count = tf.count_nonzero(converged_bitmask)
        diverged_count = tf.count_nonzero(tf.greater_equal(distance_no_nan, delta))

        next_damping = damping * tf.where(tf.less(evaluated_function_nminus * evaluated_function, tf.zeros_like(evaluated_function)), tf.ones_like(evaluated_function) * self.options.damping, tf.ones_like(evaluated_function) * self.options.speed_up)

        damping_step = damping.assign(next_damping)
        evaluation_step = tf.group(evaluated_function_nminus.assign(evaluated_function),
                                   i.assign(i + 1.0))
        ray_step = t.assign(distance_step)

        light = {"position": np.array([float(0.0), float(1.0), float(1.0)], dtype=np.float32),
                 "color": np.array([255, 255, 255], dtype=np.float32)}

        # Tensorflow has a "bug" that causes the gradient of unconnected tensors to be None instead of zeroes.
        # This will not be fixed as it is used in a lot of code to optimize and None can have different meanings.
        # Read more about it here: https://github.com/tensorflow/tensorflow/issues/783
        gradients = tf.gradients(evaluated_function, [x, y, z])
        for i, g in enumerate(gradients):
            if g is None:
                gradients[i] = tf.zeros_like(evaluated_function)
        gradient = tf.stack(gradients)

        normal_vector = tfh.normalize_vector(gradient)
        incidence = normal_vector - tfh.vector_fill(resolution, light["position"], dtype=tf.float32)
        normalized_incidence = tfh.normalize_vector(incidence)
        incidence_angle = tf.reduce_sum(normalized_incidence * normal_vector, reduction_indices=0)

        # Split the color into three channels
        light_intensity = tfh.vector_fill(resolution, light['color']) * incidence_angle

        # Add ambient light
        ambient_color = np.array([119, 139, 165], dtype=np.float32)
        with_ambient = 0.5 * light_intensity + 0.5 * tfh.vector_fill(resolution, ambient_color)
        lighted = with_ambient

        # Mask out pixels not on the surface
        bitmask = tf.less_equal(distance, epsilon)
        lighted = tf.where(tf.is_nan(lighted), tf.ones_like(lighted) * 100.0, lighted)
        masked = lighted * tf.to_float(bitmask)
        sky_color = np.array([70, 130, 180], dtype=np.float32)
        background = tfh.vector_fill(resolution, sky_color) * tf.to_float(tf.greater_equal(distance_no_nan, delta))

        # Add not converged hints
        bitmask_f = tf.logical_and(tf.greater(distance, epsilon), tf.less(distance_no_nan, delta))
        nc_color = np.array([255, 0, 0], dtype=np.float32)
        not_converged = tfh.vector_fill(resolution, nc_color) * tf.to_float(bitmask_f)

        image_data = tf.cast(masked + background + not_converged, tf.uint8)

        image = tf.transpose(image_data)
        render = tf.image.encode_jpeg(image)

        session = tf.Session()
        session.run(tf.global_variables_initializer())
        step = 0
        max_steps = self.options.max_steps
        total = resolution[0] * resolution[1]
        sys.stdout.write("\n")
        while step < max_steps:
            converged = session.run(converged_count)
            diverged = session.run(diverged_count)
            damped = session.run(damping)
            left = total - converged - diverged
            sys.stdout.write(
                "\r\033[2KStep: %s/%s, Converged: %s, Diverged: %s, Left: %s" % (step, max_steps, converged, diverged, left))
            if left == 0:
                break
            session.run(damping_step)
            session.run(evaluation_step)
            session.run(ray_step)
            step += 1
            # sys.stdout.write("")
            # sys.stdout.write("\r\033[2K")

        tracer_session = TracerSession()
        tracer_session.session = session
        tracer_session.image = image
        tracer_session.render = render

        return tracer_session


class TracerOptions:
    def __init__(self):
        self.resolution = (300, 300)
        self.camera_position = [0.0, 0.0, 1.5]
        self.v_unit = [1.0, 0.0, 0.0]
        self.lookAt = (0.0, 0.0, 0.0)
        self.verbose = True
        self.speed_up = 1.25
        self.damping = 0.8
        self.epsilon = 0.00001
        self.max_steps = 100
        self.focal_length = 1


class TracerSession:
    def __init__(self):
        self.session = None
        self.image = None
        self.render = None
