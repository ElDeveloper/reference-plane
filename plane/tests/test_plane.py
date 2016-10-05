from __future__ import division, absolute_import

import numpy as np
import pandas as pd

from unittest import main, TestCase
from io import StringIO
from skbio import OrdinationResults
from plane.plane import (point_to_plane_distance, point_to_segment_distance,
                         compute_coefficients, distance_to_reference_plane)


class TestPlane(TestCase):
    def setUp(self):
        or_f = StringIO(PCOA_STRING)
        self.ord_res = OrdinationResults.read(or_f)

        data = \
            [['PC.354', 'Control', '20061218', 'Ctrol_mouse_I.D._354'],
             ['PC.355', 'Control', '20061218', 'Control_mouse_I.D._355'],
             ['PC.356', 'Control', '20061126', 'Control_mouse_I.D._356'],
             ['PC.481', 'Control', '20070314', 'Control_mouse_I.D._481'],
             ['PC.593', 'Control', '20071210', 'Control_mouse_I.D._593'],
             ['PC.607', 'Fast', '20071112', 'Fasting_mouse_I.D._607'],
             ['PC.634', 'Fast', '20080116', 'Fasting_mouse_I.D._634'],
             ['PC.635', 'Fast', '20080116', 'Fasting_mouse_I.D._635'],
             ['PC.636', 'Fast', '20080116', 'Fasting_mouse_I.D._636']]
        headers = ['SampleID', 'Treatment', 'DOB', 'Description']
        self.mf = pd.DataFrame(data=data, columns=headers)
        self.mf.set_index('SampleID', inplace=True)

    def tearDown(self):
        pass

    def test_point_to_plane_distance(self):
        t_abcd = np.array([2, -2, 5, 8])
        t_point = np.array([4, -4, 3])

        obs = point_to_plane_distance(t_abcd, t_point)
        np.testing.assert_almost_equal(obs, 6.78902858)

        t_point = np.array([0, 0, 3])
        obs = point_to_plane_distance(t_abcd, t_point)
        np.testing.assert_almost_equal(obs, 4.00378608)

        # distance from point (2, 8, 5) to plane x-2y-2z=1
        obs = point_to_plane_distance([1, -2, -2, -1], [2, 8, 5])
        np.testing.assert_almost_equal(obs, 8.3333333333333339)

    def test_point_to_segment_distance(self):
        xyz = np.array([[54, -59, -66],
                        [41, 41, 94],
                        [62, 71, 49],
                        [77, -5, -54],
                        [-34, 37, 60],
                        [-66, 31, 20],
                        [-64, 11, 22],
                        [10, -52, -34],
                        [-93, -86, -20],
                        [99, -40, 95]])
        abcd = np.array([0.05675653, 0.62083863, -1., 19.27817089])

        point = np.array([99, -40, 95])
        obs = point_to_segment_distance(abcd, point, xyz)
        np.testing.assert_almost_equal(obs, 80.65654409263848)

        # this point lays inside the ranges of the plane, so it should give
        # the same result as with the point_to_plane_distance calculation
        point = np.zeros(3)
        obs = point_to_segment_distance(abcd, point, xyz)
        np.testing.assert_almost_equal(obs, 16.35940725554848)

        # this point lays inside the ranges of the plane, so it should give
        # the same result as with the point_to_plane_distance calculation
        point = np.ones(3) * -1
        obs = point_to_segment_distance(abcd, point, xyz)
        np.testing.assert_almost_equal(obs, 16.63299919102415)

    def test_compute_coefficients(self):
        xyz = np.array([[54, -59, -66],
                        [41, 41, 94],
                        [62, 71, 49],
                        [77, -5, -54],
                        [-34, 37, 60],
                        [-66, 31, 20],
                        [-64, 11, 22],
                        [10, -52, -34],
                        [-93, -86, -20],
                        [99, -40, 95]])
        obs = compute_coefficients(xyz)
        np.testing.assert_almost_equal(obs, np.array([0.0567565, 0.6208386, -1,
                                                      19.2781709]))

    def test_distance_to_reference_plane(self):
        obs = distance_to_reference_plane(self.ord_res, self.mf, 'Control',
                                          'Treatment')

        index = ['PC.636', 'PC.635', 'PC.356', 'PC.481', 'PC.354', 'PC.593',
                 'PC.355', 'PC.607', 'PC.634']
        vals = [0.34417253293, 0.348491435304, 0.0935726193327,
                0.0288092804881, 0.00541912134103, 0.0617752337758,
                0.0900316761388, 0.519899530142, 0.420389532328]
        exp = pd.Series(vals, index=index)

        pd.util.testing.assert_series_equal(obs, exp)

    def test_reference_plane_bad_arguments(self):
        with self.assertRaises(ValueError):
            distance_to_reference_plane(self.ord_res, self.mf, 'Control')

    def test_reference_plane_no_matches(self):
        new = self.mf.copy()

        # no values will match these integers
        new.index = pd.Index(np.arange(9).astype('str'))

        with self.assertRaises(KeyError):
            distance_to_reference_plane(self.ord_res, new, 'Control',
                                        'Treatment')

    def test_reference_plane_non_existant_category(self):
        with self.assertRaises(ValueError):
            distance_to_reference_plane(self.ord_res, self.mf,
                                        'Does not exist', 'Treatment')


if __name__ == '__main__':
    main()

PCOA_STRING = u"""Eigvals\t9
0.479412119045\t0.29201495623\t0.247449246064\t0.201496072404\t0.180076127632\
\t0.147806772727\t0.135795927213\t0.112259695609\t0.0

Proportion explained\t9
0.266887048633\t0.162563704022\t0.137754129161\t0.11217215823\t0.10024774995\
\t0.0822835130237\t0.0755971173665\t0.0624945796136\t0.0

Species\t0\t0

Site\t9\t9
PC.636\t-0.276542163845\t-0.144964375408\t0.0666467344429\t-0.0677109454288\
\t0.176070269506\t0.072969390136\t-0.229889463523\t-0.0465989416581\
\t-0.0
PC.635\t-0.237661393984\t0.0460527772512\t-0.138135814766\t0.159061025229\
\t-0.247484698646\t-0.115211468101\t-0.112864033263\t0.0647940729676\
\t-0.0
PC.356\t0.228820399536\t-0.130142097093\t-0.287149447883\t0.0864498846421\
\t0.0442951919304\t0.20604260722\t0.0310003571386\t0.0719920436501\t-0.0
PC.481\t0.0422628480532\t-0.0139681511889\t0.0635314615517\t-0.346120552134\
\t-0.127813807608\t0.0139350721063\t0.0300206887328\t0.140147849223\t-0.0
PC.354\t0.280399117569\t-0.0060128286014\t0.0234854344148\t-0.0468109474823\
\t-0.146624450094\t0.00566979124596\t-0.0354299634191\
\t-0.255785794275\t-0.0
PC.593\t0.232872767451\t0.139788385269\t0.322871079774\t0.18334700682\
\t0.0204661596818\t0.0540589147147\t-0.0366250872041\t0.0998235721267\
\t-0.0
PC.355\t0.170517581885\t-0.194113268955\t-0.0308965283066\t0.0198086158783\
\t0.155100062794\t-0.279923941712\t0.0576092515759\t0.0242481862127\t-0.0
PC.607\t-0.0913299284215\t0.424147148265\t-0.135627421345\t-0.057519480907\
\t0.151363490722\t-0.0253935675552\t0.0517306152066\t-0.038738217609\
\t-0.0
PC.634\t-0.349339228244\t-0.120787589539\t0.115274502117\t0.0694953933826\
\t-0.0253722182853\t0.067853201946\t0.244447634756\t-0.0598827706386\
\t-0.0

Biplot\t0\t0

Site constraints\t0\t0
"""
