import re
import os
import json
import glob
import logging
import itertools
import numpy as np
import pydicom as dicom
import matplotlib.pyplot as plt

logger = logging.getLogger()

def normalize(x):
  return x / np.sqrt(np.dot(x,x))

def readRotAndTransFromDicom(paths):
  failure = re.compile(r'^.*\? F:')
  pattern = re.compile(r'.*R:\s+(.*?)\s+(.*?)\s+(.*?)\s+(.*?)\s+T:\s+(.*?)\s+(.*?)\s+(.*?)\s+F:.*')

  files = itertools.chain.from_iterable([glob.glob(path) for path in paths])

  # Sort DICOMs by AcquisitionNumber (important!)
  ds = sorted([dicom.dcmread(x) for x in files], key=lambda dcm: dcm.AcquisitionNumber)

  failed = None

  # Return the "raw" quaternion values stored in the DICOM ImageComments field
  quaternions = [(
    np.array([1,0,0,0]),
    np.array([0,0,0])
  )]
  for x in ds[1:]:
    instance = x.InstanceNumber
    acquisition = x.AcquisitionNumber
    '''
    vNav failures do not appear to be encoded consistently. The following regular
    expression was crafted based on a small number of examples.
    '''
    if 'ImageComments' not in x or failure.match(x.ImageComments):
        logger.error(f'failure detected instance={instance}, acquisition={acquisition}')
        failed = (instance, acquisition)
        break
    match = pattern.match(x.ImageComments)
    if not match:
        logger.error(f'ImageComments from instance={instance}, acquisition={acquisition} failed pattern match')
        logger.error(x.ImageComments)
        logger.error(f'treating as vNav failure')
        failed = (instance, acquisition)
        break

    R = match.groups(0)[0:4]
    T = match.groups(0)[4:]
    logger.debug(x.ImageComments)
    logger.debug(f'R={R} and T={T}')
    rotation = np.array(list(map(float, R)))
    translation = np.array(list(map(float, T)))
    quaternions.append((rotation, translation))

  return quaternions,failed

def readRotAndTransFromJson(js):
  '''
  vNav failures do not appear to be encoded consistently. The following regular
  expression was crafted based on a small number of examples.
  '''
  failure = r'^.*\?_F:'
  pattern = r'R:_(.*?)_(.*?)_(.*?)_(.*?)_T:_(.*?)_(.*?)_(.*?)_F:.'
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
  t_rotmax = radius * np.sqrt(2.0 - 2.0 * np.cos(angle))
  return np.sqrt(
    (t_rotmax * t_rotmax) +
    (2.0 * t_rotmax) *
      np.linalg.norm(
        trans - (np.dot(trans, axis) * axis)) +
    (np.linalg.norm(trans) * np.linalg.norm(trans))
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

