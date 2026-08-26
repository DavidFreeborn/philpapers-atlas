import assert from 'node:assert/strict';
import test from 'node:test';

import {
  DEFAULT_CAMERA_3D,
  MAX_CAMERA_DISTANCE,
  MAX_CAMERA_PITCH,
  MIN_CAMERA_DISTANCE,
  cameraPosition,
  effectiveVerticalFov,
  fitCameraToSphere,
  orbitCamera,
  panCamera,
  projectPoint,
  resetCamera3D,
  viewProjectionMatrix,
  zoomCamera,
} from '../app/atlas3d.ts';

const close = (left: number, right: number, tolerance = 1e-5) => {
  assert.ok(Math.abs(left - right) <= tolerance, `${left} is not close to ${right}`);
};

test('the camera target projects to the screen centre', () => {
  const camera = resetCamera3D();
  const projected = projectPoint(camera.target, viewProjectionMatrix(camera, 16 / 9), 1600, 900);
  assert.equal(projected.visible, true);
  close(projected.x, 800, 1e-3);
  close(projected.y, 450, 1e-3);
});

test('orbiting preserves target and distance and clamps pitch', () => {
  const orbit = orbitCamera(resetCamera3D(), 420, 100_000);
  assert.deepEqual(orbit.target, DEFAULT_CAMERA_3D.target);
  close(orbit.distance, DEFAULT_CAMERA_3D.distance);
  close(orbit.pitch, MAX_CAMERA_PITCH);
  close(Math.hypot(...cameraPosition(orbit).map((value, index) => value - orbit.target[index])), orbit.distance);
});

test('zoom stays within safe clipping bounds', () => {
  assert.equal(zoomCamera(resetCamera3D(), 0).distance, MIN_CAMERA_DISTANCE);
  assert.equal(zoomCamera(resetCamera3D(), 1e6).distance, MAX_CAMERA_DISTANCE);
});

test('panning changes the target without changing the orbit', () => {
  const camera = resetCamera3D();
  const panned = panCamera(camera, 120, -50, 900, 16 / 9);
  assert.notDeepEqual(panned.target, camera.target);
  close(panned.distance, camera.distance);
  close(panned.yaw, camera.yaw);
  close(panned.pitch, camera.pitch);
});

test('portrait projection widens its vertical field to preserve horizontal framing', () => {
  assert.ok(effectiveVerticalFov(0.5) > effectiveVerticalFov(1.5));
});

test('fitting a sphere centres it and increases distance with radius', () => {
  const small = fitCameraToSphere(resetCamera3D(), [1, 2, 3], 0.1, 1.5);
  const large = fitCameraToSphere(resetCamera3D(), [1, 2, 3], 0.8, 1.5);
  assert.deepEqual(small.target, [1, 2, 3]);
  assert.ok(large.distance > small.distance);
});

test('points behind the camera are rejected', () => {
  const camera = resetCamera3D();
  const position = cameraPosition(camera);
  const behind: [number, number, number] = [
    position[0] * 2 - camera.target[0],
    position[1] * 2 - camera.target[1],
    position[2] * 2 - camera.target[2],
  ];
  const projected = projectPoint(behind, viewProjectionMatrix(camera, 1), 800, 800);
  assert.equal(projected.visible, false);
});
