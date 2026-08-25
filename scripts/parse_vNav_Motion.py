#!/usr/bin/env python

import os
import vnav
import json
import logging
import argparse
import numpy as np
import pydicom as dicom
from argparse import ArgumentParser

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

def main():
    parser = ArgumentParser()

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

    if args.verbose:
      logger.setLevel(logging.DEBUG)

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
          logger.warning(f'ignoring invalid dicom {f}')
          continue
        args.input.append(f)

    if args.input_json:
      args.input_json = os.path.expanduser(args.input_json)
      with open(args.input_json, 'r') as fo:
        args.input_json = json.load(fo)

    ## Perform calcs and generate output
    if args.input:
      transforms,err = vnav.readRotAndTransFromDicom(args.input)
    elif args.input_json:
      transforms,err = vnav.readRotAndTransFromJson(args.input_json)

    summary['Transforms'] = [(rot.tolist(), tra.tolist()) for (rot, tra) in transforms]

    transforms = [vnav.motionEntryToHomogeneousTransform(e) for e in transforms]

    summary['HomogeneousTransforms'] = [x.tolist() for x in transforms]

    summary['Failed'] = None
    if err:
        summary['Failed'] = {
            'Instance': err[0],
            'Acquisition': err[1]
        }

    diffTransforms = [ts[1] * np.linalg.inv(ts[0]) for ts in zip(transforms[0:], transforms[1:])]

    rmsMotionScores = [vnav.diffTransformToRMSMotion(t, args.radius) for t in diffTransforms]

    maxMotionScores = [vnav.diffTransformToMaxMotion(t, args.radius) for t in diffTransforms]

    if args.rms:
      MeanMotionScoreRMSPerMin = np.mean(rmsMotionScores) * 60.0 / float(args.tr)
      summary['vNavMotionScoresRMS'] = rmsMotionScores
      summary['MeanMotionScoreRMSPerMin'] = MeanMotionScoreRMSPerMin
      print('MeanMotionScoreRMSPerMin:', MeanMotionScoreRMSPerMin)
      if args.plot:
        vnav.simplePlot(rmsMotionScores, 'RMS', args.radius, args.tr, args.output_dir)

    if args.max:
      MeanMotionScoreMaxPerMin = np.mean(maxMotionScores) * 60.0 / float(args.tr)
      summary['vNavMotionScoresMax'] = maxMotionScores
      summary['MeanMotionScoreMaxPerMin'] = MeanMotionScoreMaxPerMin
      print('MeanMotionScoreMaxPerMin:', MeanMotionScoreMaxPerMin)
      if args.plot:
        vnav.simplePlot(maxMotionScores, 'Max', args.radius, args.tr, args.output_dir)

    if summary:
      filename = os.path.join(args.output_dir, 'vNav_Motion.json')
      with open(filename, 'w') as fo:
        fo.write(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()

