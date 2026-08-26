export type Vec3 = [number, number, number];

export type Camera3D = {
  yaw: number;
  pitch: number;
  distance: number;
  target: Vec3;
};

export type ProjectedPoint = {
  x: number;
  y: number;
  depth: number;
  visible: boolean;
};

export const DEFAULT_CAMERA_3D: Camera3D = {
  yaw: 0.36,
  pitch: 0.24,
  distance: 3.25,
  target: [0, 0, 0],
};

export const MIN_CAMERA_DISTANCE = 0.16;
export const MAX_CAMERA_DISTANCE = 12;
export const MAX_CAMERA_PITCH = Math.PI / 2 - 0.055;
const BASE_VERTICAL_FOV = 46 * Math.PI / 180;

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function subtract(left: Vec3, right: Vec3): Vec3 {
  return [left[0] - right[0], left[1] - right[1], left[2] - right[2]];
}

function cross(left: Vec3, right: Vec3): Vec3 {
  return [
    left[1] * right[2] - left[2] * right[1],
    left[2] * right[0] - left[0] * right[2],
    left[0] * right[1] - left[1] * right[0],
  ];
}

function dot(left: Vec3, right: Vec3): number {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

function normalize(value: Vec3): Vec3 {
  const length = Math.hypot(value[0], value[1], value[2]) || 1;
  return [value[0] / length, value[1] / length, value[2] / length];
}

export function effectiveVerticalFov(aspect: number): number {
  const safeAspect = Math.max(aspect, 0.05);
  return safeAspect >= 1
    ? BASE_VERTICAL_FOV
    : 2 * Math.atan(Math.tan(BASE_VERTICAL_FOV / 2) / safeAspect);
}

export function cameraPosition(camera: Camera3D): Vec3 {
  const horizontal = Math.cos(camera.pitch) * camera.distance;
  return [
    camera.target[0] + Math.sin(camera.yaw) * horizontal,
    camera.target[1] + Math.sin(camera.pitch) * camera.distance,
    camera.target[2] + Math.cos(camera.yaw) * horizontal,
  ];
}

export function orbitCamera(camera: Camera3D, deltaX: number, deltaY: number): Camera3D {
  return {
    ...camera,
    yaw: camera.yaw - deltaX * 0.006,
    pitch: clamp(camera.pitch + deltaY * 0.006, -MAX_CAMERA_PITCH, MAX_CAMERA_PITCH),
  };
}

export function zoomCamera(camera: Camera3D, factor: number): Camera3D {
  return {
    ...camera,
    distance: clamp(camera.distance * factor, MIN_CAMERA_DISTANCE, MAX_CAMERA_DISTANCE),
  };
}

export function panCamera(
  camera: Camera3D,
  deltaX: number,
  deltaY: number,
  viewportHeight: number,
  aspect: number,
): Camera3D {
  const position = cameraPosition(camera);
  const forward = normalize(subtract(camera.target, position));
  const worldUp: Vec3 = [0, 1, 0];
  let right = normalize(cross(forward, worldUp));
  if (Math.abs(dot(forward, worldUp)) > 0.995) right = [Math.cos(camera.yaw), 0, -Math.sin(camera.yaw)];
  const up = normalize(cross(right, forward));
  const worldPerPixel = (
    2 * camera.distance * Math.tan(effectiveVerticalFov(aspect) / 2)
  ) / Math.max(viewportHeight, 1);
  return {
    ...camera,
    target: [
      camera.target[0] - right[0] * deltaX * worldPerPixel + up[0] * deltaY * worldPerPixel,
      camera.target[1] - right[1] * deltaX * worldPerPixel + up[1] * deltaY * worldPerPixel,
      camera.target[2] - right[2] * deltaX * worldPerPixel + up[2] * deltaY * worldPerPixel,
    ],
  };
}

export function perspectiveMatrix(aspect: number, near = 0.025, far = 40): Float32Array {
  const f = 1 / Math.tan(effectiveVerticalFov(aspect) / 2);
  const range = 1 / (near - far);
  return new Float32Array([
    f / Math.max(aspect, 0.05), 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * range, -1,
    0, 0, 2 * far * near * range, 0,
  ]);
}

export function lookAtMatrix(position: Vec3, target: Vec3): Float32Array {
  const zAxis = normalize(subtract(position, target));
  let xAxis = normalize(cross([0, 1, 0], zAxis));
  if (Math.hypot(xAxis[0], xAxis[1], xAxis[2]) < 1e-6) xAxis = [1, 0, 0];
  const yAxis = cross(zAxis, xAxis);
  return new Float32Array([
    xAxis[0], yAxis[0], zAxis[0], 0,
    xAxis[1], yAxis[1], zAxis[1], 0,
    xAxis[2], yAxis[2], zAxis[2], 0,
    -dot(xAxis, position), -dot(yAxis, position), -dot(zAxis, position), 1,
  ]);
}

export function multiplyMatrices(left: Float32Array, right: Float32Array): Float32Array {
  const result = new Float32Array(16);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      let value = 0;
      for (let index = 0; index < 4; index += 1) {
        value += left[index * 4 + row] * right[column * 4 + index];
      }
      result[column * 4 + row] = value;
    }
  }
  return result;
}

export function viewProjectionMatrix(camera: Camera3D, aspect: number): Float32Array {
  return multiplyMatrices(
    perspectiveMatrix(aspect),
    lookAtMatrix(cameraPosition(camera), camera.target),
  );
}

export function projectPoint(
  point: Vec3,
  matrix: Float32Array,
  width: number,
  height: number,
): ProjectedPoint {
  const x = point[0];
  const y = point[1];
  const z = point[2];
  const clipX = matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12];
  const clipY = matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13];
  const clipZ = matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14];
  const clipW = matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15];
  if (clipW <= 0) return { x: 0, y: 0, depth: clipW, visible: false };
  const ndcX = clipX / clipW;
  const ndcY = clipY / clipW;
  const ndcZ = clipZ / clipW;
  return {
    x: (ndcX * 0.5 + 0.5) * width,
    y: (0.5 - ndcY * 0.5) * height,
    depth: clipW,
    visible: ndcZ >= -1 && ndcZ <= 1 && ndcX >= -1.08 && ndcX <= 1.08 && ndcY >= -1.08 && ndcY <= 1.08,
  };
}

export function fitCameraToSphere(
  camera: Camera3D,
  center: Vec3,
  radius: number,
  aspect: number,
): Camera3D {
  const verticalFov = effectiveVerticalFov(aspect);
  const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * Math.max(aspect, 0.05));
  const limitingFov = Math.min(verticalFov, horizontalFov);
  const safeRadius = Math.max(radius, 0.015);
  return {
    ...camera,
    target: [...center],
    distance: clamp((safeRadius / Math.sin(limitingFov / 2)) * 1.16, MIN_CAMERA_DISTANCE, MAX_CAMERA_DISTANCE),
  };
}

export function resetCamera3D(): Camera3D {
  return {
    ...DEFAULT_CAMERA_3D,
    target: [...DEFAULT_CAMERA_3D.target],
  };
}
