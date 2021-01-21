#!/usr/bin/env python

import pydicom as dicom
import os
import numpy as np
import itertools
import glob
import argparse
import json
import re
import matplotlib.pyplot as plt

def normalize(x):
  return x / np.sqrt(np.dot(x,x))

def readRotAndTransFromDicom(paths):
  files = itertools.chain.from_iterable([glob.glob(path) for path in paths])

  # Sort DICOMs by AcquisitionNumber (important!)
  ds = sorted([dicom.read_file(x) for x in files], key=lambda dcm: dcm.AcquisitionNumber)

  failed = None

  # Return the "raw" quaternion values stored in the DICOM ImageComments field
  quaternions = [(
    np.array([1,0,0,0]),
    np.array([0,0,0])
  )]
  for x in ds[1:]:
    instance = x.InstanceNumber
    acquisition = x.AcquisitionNumber
    # check if vNav failed
    if re.match('^\? F:', x.ImageComments):
        print(f'failure detected instance={instance}, acquisition={acquisition}')
        failed = (instance, acquisition)
        break
    y = str.split(x.ImageComments)
    rotation = np.array(list(map(float, y[1:5])))
    translation = np.array(list(map(float, y[6:9])))
    quaternions.append((rotation, translation))

  return quaternions,failed

def readRotAndTransFromJson(js):
  failure = '^\?_F:'
  pattern = 'R:_(.*?)_(.*?)_(.*?)_(.*?)_T:_(.*?)_(.*?)_(.*?)_F:.'
  failed = None
  # parse ImageComments and return quaternion values
  quaternions = [(
    np.array([1,0,0,0]),
    np.array([0,0,0])
  )]
  for i,comment in enumerate(js['ImageComments'][1:]):
    if re.match(failure, comment):
      acquisition = js['AcquisitionNumber'][i]
      failed = (None, acquisition)
      break
    m = re.match(pattern, comment)
    rotation = np.array(list(map(float, m.groups(0)[0:4])))
    translation = np.array(list(map(float, m.groups(0)[4:])))
    quaternions.append((rotation,translation))

  return quaternions,failed

def angleAxisToQuaternion(a):
  w = np.cos(a[0] / 2.0)
  axisNorm = np.sqrt(np.dot(a[1:], a[1:]))

  if 0 == axisNorm :
    return np.array([1,0,0,0])

  axisScale = np.sin(a[0] / 2.0) / np.sqrt(np.dot(a[1:], a[1:]))
  tail = a[1:] * axisScale
  q = np.ndarray(shape=(4))
  q[0] = w
  q[1:] = tail
  return q

def quaternionToAxisAngle(q) :
  a = np.ndarray(shape=(4))
  a[0] = np.arccos(q[0]) * 2.0
  a[1:] = q[1:] / np.sqrt(np.dot(q[1:], q[1:]))
  return a

def quaternionToRotationMatrix(q):
  w = q[0];
  x = q[1];
  y = q[2];
  z = q[3];

  wSq = w * w;
  xSq = x * x;
  ySq = y * y;
  zSq = z * z;

  wx2 = w*x*2.0;
  wy2 = w*y*2.0;
  wz2 = w*z*2.0;

  xy2 = x*y*2.0;
  xz2 = x*z*2.0;

  yz2 = y*z*2.0;

  rotMatrix = np.ndarray(shape=(3,3))

  rotMatrix[0,0] = wSq + xSq - ySq - zSq;
  rotMatrix[0,1] = xy2 - wz2;
  rotMatrix[0,2] = xz2 + wy2;

  rotMatrix[1,0] = xy2 + wz2;
  rotMatrix[1,1] = wSq - xSq + ySq - zSq;
  rotMatrix[1,2] = yz2 - wx2;

  rotMatrix[2,0] = xz2 - wy2;
  rotMatrix[2,1] = yz2 + wx2;
  rotMatrix[2,2] = wSq - xSq - ySq + zSq;

  return rotMatrix

def rotationMatrixToQuaternion(m):
  ## Dylan Dec 19/12
  ## This algorithm taken from http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
  ##
  ## Also tried the algorithm in
  ##		Animating rotation with quaternion curves.
  ##		Ken Shoemake
  ##		Computer Graphics 19(3):245-254,  1985
  ##		http://portal.acm.org/citation.cfm?doid=325334.325242
  ## but you'll find that it's not numerically stable without some truncation epsilon.
  ## The algorithm we're using now doesn't require us to pick some arbitrary epsilon, so
  ## I like it better.

  tr = np.trace(m);

  if (tr > 0) :
    S = np.sqrt(tr+1.0) * 2; # S=4*qw
    SInv = 1.0 / S;
    w = 0.25 * S;
    x = (m[2,1] - m[1,2]) * SInv;
    y = (m[0,2] - m[2,0]) * SInv;
    z = (m[1,0] - m[0,1]) * SInv;
  elif ((m[0,0] > m[1,1]) and (m[0,0] > m[2,2])):
    S = np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2; # S=4*qx
    SInv = 1.0 / S;
    w = (m[2,1] - m[1,2]) * SInv;
    x = 0.25 * S;
    y = (m[0,1] + m[1,0]) * SInv;
    z = (m[0,2] + m[2,0]) * SInv
  elif (m[1,1] > m[2,2]):
    S = np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2; # S=4*qy
    SInv = 1.0 / S;
    w = (m[0,2] - m[2,0]) * SInv;
    x = (m[0,1] + m[1,0]) * SInv;
    y = 0.25 * S;
    z = (m[1,2] + m[2,1]) * SInv;
  else:
    S = np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2; # S=4*qz
    SInv = 1.0 / S;
    w = (m[1,0] - m[0,1]) * SInv;
    x = (m[0,2] + m[2,0]) * SInv;
    y = (m[1,2] + m[2,1]) * SInv;
    z = 0.25 * S;

  return np.array([w, x, y, z])

def motionEntryToHomogeneousTransform(e) :
  t = np.ndarray(shape=(4,4))

  t[0:3,3] = e[1]
  t[0:3,0:3] = quaternionToRotationMatrix(angleAxisToQuaternion(e[0]))
  t[3,:] = [0,0,0,1]
  return np.matrix(t)

def diffTransformToMaxMotion(t, radius):
  angleAxis = quaternionToAxisAngle(rotationMatrixToQuaternion(t[0:3, 0:3]))
  angle = angleAxis[0]
  axis = angleAxis[1:]
  trans = t[0:3,3].flatten()
  t_rotmax = radius * np.sqrt(2 - 2 * np.cos(angle))
  return np.sqrt(
    t_rotmax * t_rotmax +
    2 * t_rotmax *
      np.linalg.norm(
        trans - (np.dot(trans, axis) * axis)) +
    np.linalg.norm(trans) * np.linalg.norm(trans)
    )

def diffTransformToRMSMotion(t, radius):
  rotMatMinusIdentity = t[0:3,0:3] - np.array([[1,0,0],[0,1,0],[0,0,1]])
  trans = np.ravel(t[0:3,3])

  return np.sqrt(
    0.2 * radius * radius * np.trace(np.transpose(rotMatMinusIdentity) * rotMatMinusIdentity) +
    np.dot(trans, trans)
    )

def simplePlot(scores, measStr, radius, TR, output_dir):
  fig, ax = plt.subplots()
  ax.plot(scores)
  xlabel = 'Frame (TR=' + str(TR) + ' s)'
  ylabel = 'mm (assumed radius=' + str(radius) + ' mm)'
  titleStr = 'vNavMotionScores (measure=' + measStr + ')'
  ax.set(xlabel=xlabel, ylabel=ylabel, title=titleStr)
  filename = os.path.join(output_dir, 'vNavMotionScores'+measStr+'.png')
  fig.savefig(filename)


parser = argparse.ArgumentParser()

# Required arguments
parser.add_argument('--tr', type=float, required=True, help='TR (sec)')
inputs = parser.add_mutually_exclusive_group(required=True)
inputs.add_argument('--input', nargs='+', help='List of input DICOMs')
inputs.add_argument('--input-dir', help='Directory of DICOMs')
inputs.add_argument('--input-json', help='Input BIDS JSON sidecar')
output_type = parser.add_argument_group('measure (at least one required)')
output_type.add_argument('--rms', action='store_true')
output_type.add_argument('--max', action='store_true')
# Optional arguments
parser.add_argument('--radius', type=float, default=100, help='radius (mm) of assumed sphere (default: %(default)s)')
parser.add_argument('--plot', help='output plot of chosen measure across frames', action='store_true')
parser.add_argument('-v','--verbose', help='increase output verbosity', action='store_true')
parser.add_argument('--output-dir', default='.', help='output directory')

args = parser.parse_args()

if args.rms is False and args.max is False:
  parser.error('At least one of --rms or --max is required.')

if not os.path.exists(args.output_dir):
  os.makedirs(args.output_dir)

summary = dict()

if args.input_dir:
  args.input_dir = os.path.expanduser(args.input_dir)
  args.input = list()
  for f in os.listdir(args.input_dir):
    f = os.path.join(args.input_dir, f)
    # only add valid dicom files from input directory
    try:
      with open(f, 'rb') as fo:
        dicom.filereader.read_preamble(fo, force=False)
    except dicom.errors.InvalidDicomError:
      print(f'ignoring invalid dicom {f}')
      continue
    args.input.append(f)

if args.input_json:
  args.input_json = os.path.expanduser(args.input_json)
  with open(args.input_json, 'r') as fo:
    args.input_json = json.load(fo)

## Perform calcs and generate output
if args.input:
  transforms,err = readRotAndTransFromDicom(args.input)
elif args.input_json:
  transforms,err = readRotAndTransFromJson(args.input_json)

summary['Transforms'] = [(rot.tolist(), tra.tolist()) for (rot, tra) in transforms]

transforms = [motionEntryToHomogeneousTransform(e) for e in transforms]

summary['HomogeneousTransforms'] = [x.tolist() for x in transforms]

summary['Failed'] = None
if err:
    summary['Failed'] = {
        'Instance': err[0],
        'Acquisition': err[1]
    }

diffTransforms = [ts[1] * np.linalg.inv(ts[0]) for ts in zip(transforms[0:], transforms[1:])]

rmsMotionScores = [diffTransformToRMSMotion(t, args.radius) for t in diffTransforms]

maxMotionScores = [diffTransformToMaxMotion(t, args.radius) for t in diffTransforms]

if args.rms:
  MeanMotionScoreRMSPerMin = np.mean(rmsMotionScores) * 60.0 / float(args.tr)
  summary['vNavMotionScoresRMS'] = rmsMotionScores
  summary['MeanMotionScoreRMSPerMin'] = MeanMotionScoreRMSPerMin
  print('MeanMotionScoreRMSPerMin:', MeanMotionScoreRMSPerMin)
  if args.plot:
    simplePlot(rmsMotionScores, 'RMS', args.radius, args.tr, args.output_dir)

if args.max:
  MeanMotionScoreMaxPerMin = np.mean(maxMotionScores) * 60.0 / float(args.tr)
  summary['vNavMotionScoresMax'] = maxMotionScores
  summary['MeanMotionScoreMaxPerMin'] = MeanMotionScoreMaxPerMin
  print('MeanMotionScoreMaxPerMin:', MeanMotionScoreMaxPerMin)
  if args.plot:
    simplePlot(maxMotionScores, 'Max', args.radius, args.tr, args.output_dir)

if summary:
  filename = os.path.join(args.output_dir, 'vNav_Motion.json')
  with open(filename, 'w') as fo:
    fo.write(json.dumps(summary, indent=2))

