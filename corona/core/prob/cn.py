""" Predefined Probability Tables for valuation in China
"""
from .prob import Probability


__all__ = ['CL03', 'CL03A', 'CL13_I', 'CL13_II', 'CL13A']


CL03 = Probability([
    [0.000722, 0.000603, 0.000499, 0.000416, 0.000358, 0.000323, 0.000309, 0.000308, 0.000311,
     0.000312, 0.000312, 0.000312, 0.000313, 0.00032, 0.000336, 0.000364, 0.000404, 0.000455,
     0.000513, 0.000572, 0.000621, 0.000661, 0.000692, 0.000716, 0.000738, 0.000759, 0.000779,
     0.000795, 0.000815, 0.000842, 0.000881, 0.000932, 0.000994, 0.001055, 0.001121, 0.001194,
     0.001275, 0.001367, 0.001472, 0.001589, 0.001715, 0.001845, 0.001978, 0.002113, 0.002255,
     0.002413, 0.002595, 0.002805, 0.003042, 0.003299, 0.00357, 0.003847, 0.004132, 0.004434,
     0.004778, 0.005203, 0.005744, 0.006427, 0.00726, 0.008229, 0.009313, 0.01049, 0.011747,
     0.013091, 0.014542, 0.016134, 0.017905, 0.019886, 0.022103, 0.024571, 0.027309, 0.03034,
     0.033684, 0.037371, 0.04143, 0.045902, 0.050829, 0.056262, 0.062257, 0.068871, 0.076187,
     0.084224, 0.093071, 0.1028, 0.113489, 0.125221, 0.13808, 0.152157, 0.167543, 0.184333,
     0.202621, 0.2225, 0.244059, 0.267383, 0.292544, 0.319604, 0.348606, 0.379572, 0.412495,
     0.447334, 0.48401, 0.522397, 0.562317, 0.603539, 0.64577, 1],
    [0.000661, 0.000536, 0.000424, 0.000333, 0.000267, 0.000224, 0.000201, 0.000189, 0.000181,
     0.000175, 0.000169, 0.000165, 0.000165, 0.000169, 0.000179, 0.000192, 0.000208, 0.000226,
     0.000245, 0.000264, 0.000283, 0.0003, 0.000315, 0.000328, 0.000338, 0.000347, 0.000355,
     0.000362, 0.000372, 0.000386, 0.000406, 0.000432, 0.000465, 0.000496, 0.000528, 0.000563,
     0.000601, 0.000646, 0.000699, 0.000761, 0.000828, 0.000897, 0.000966, 0.001033, 0.001103,
     0.001181, 0.001274, 0.001389, 0.001527, 0.00169, 0.001873, 0.002074, 0.002295, 0.002546,
     0.002836, 0.003178, 0.003577, 0.004036, 0.004556, 0.005133, 0.005768, 0.006465, 0.007235,
     0.008094, 0.009059, 0.010148, 0.011376, 0.01276, 0.014316, 0.016066, 0.018033, 0.020241,
     0.022715, 0.025479, 0.028561, 0.031989, 0.035796, 0.040026, 0.044726, 0.049954, 0.055774,
     0.062253, 0.069494, 0.077511, 0.086415, 0.096294, 0.107243, 0.119364, 0.132763, 0.147553,
     0.16385, 0.181775, 0.201447, 0.222987, 0.246507, 0.272115, 0.299903, 0.329942, 0.362281,
     0.396933, 0.433869, 0.473008, 0.514211, 0.557269, 0.601896, 1]], name='CL03')


CL03A = Probability([
    [0.000627, 0.000525, 0.000434, 0.000362, 0.000311, 0.000281, 0.000269, 0.000268, 0.00027, 0.000271,
     0.000272, 0.000271, 0.000272, 0.000278, 0.000292, 0.000316, 0.000351, 0.000396, 0.000446, 0.000497,
     0.00054, 0.000575, 0.000601, 0.000623, 0.000643, 0.00066, 0.000676, 0.000693, 0.000712, 0.000734,
     0.000759, 0.000788, 0.00082, 0.000855, 0.000893, 0.000936, 0.000985, 0.001043, 0.001111, 0.001189,
     0.001275, 0.001366, 0.001461, 0.00156, 0.001665, 0.001783, 0.001918, 0.002055, 0.002238, 0.002446,
     0.002666, 0.00288, 0.003085, 0.0033, 0.003545, 0.003838, 0.004207, 0.004676, 0.005275, 0.006039,
     0.006989, 0.007867, 0.008725, 0.009677, 0.010731, 0.0119, 0.013229, 0.014705, 0.016344, 0.018164,
     0.020184, 0.022425, 0.024911, 0.027668, 0.030647, 0.033939, 0.037577, 0.041594, 0.046028, 0.05092,
     0.056312, 0.062253, 0.068791, 0.075983, 0.083883, 0.092554, 0.102059, 0.112464, 0.123836, 0.136246,
     0.149763, 0.164456, 0.180392, 0.197631, 0.216228, 0.236229, 0.257666, 0.280553, 0.304887, 0.330638,
     0.357746, 0.386119, 0.415626, 0.446094, 0.477308, 1],
    [0.000575, 0.000466, 0.000369, 0.00029, 0.000232, 0.000195, 0.000175, 0.000164, 0.000158, 0.000152,
     0.000147, 0.000143, 0.000143, 0.000147, 0.000156, 0.000167, 0.000181, 0.000196, 0.000213, 0.00023,
     0.000246, 0.000261, 0.000274, 0.000285, 0.000293, 0.000301, 0.000308, 0.000316, 0.000325, 0.000337,
     0.000351, 0.000366, 0.000384, 0.000402, 0.000421, 0.000441, 0.000464, 0.000493, 0.000528, 0.000569,
     0.000615, 0.000664, 0.000714, 0.000763, 0.000815, 0.000873, 0.000942, 0.001014, 0.001123, 0.001251,
     0.001393, 0.001548, 0.001714, 0.001893, 0.002093, 0.002318, 0.002607, 0.002979, 0.00341, 0.003816,
     0.004272, 0.004781, 0.005351, 0.005988, 0.006701, 0.007499, 0.008408, 0.009438, 0.010592, 0.011886,
     0.013337, 0.014964, 0.016787, 0.018829, 0.021117, 0.023702, 0.026491, 0.029602, 0.03307, 0.036935,
     0.041241, 0.046033, 0.051365, 0.057291, 0.063872, 0.071174, 0.079267, 0.088225, 0.098129, 0.109061,
     0.121107, 0.134355, 0.148896, 0.164816, 0.182201, 0.201129, 0.221667, 0.24387, 0.267773, 0.293385,
     0.320685, 0.349615, 0.380069, 0.411894, 0.444879, 1]], name='CL03A')


CL13_I = Probability([
    [0.000867, 0.000615, 0.000445, 0.000339, 0.00028, 0.000251, 0.000237, 0.000233, 0.000238, 0.00025,
     0.000269, 0.000293, 0.000319, 0.000347, 0.000375, 0.000402, 0.000427, 0.000449, 0.000469, 0.000489,
     0.000508, 0.000527, 0.000547, 0.000568, 0.000591, 0.000615, 0.000644, 0.000675, 0.000711, 0.000751,
     0.000797, 0.000847, 0.000903, 0.000966, 0.001035, 0.001111, 0.001196, 0.00129, 0.001395, 0.001515,
     0.001651, 0.001804, 0.001978, 0.002173, 0.002393, 0.002639, 0.002913, 0.003213, 0.003538, 0.003884,
     0.004249, 0.004633, 0.005032, 0.005445, 0.005869, 0.006302, 0.006747, 0.007227, 0.00777, 0.008403,
     0.009161, 0.010065, 0.011129, 0.01236, 0.013771, 0.015379, 0.017212, 0.019304, 0.021691, 0.024411,
     0.027495, 0.030965, 0.034832, 0.039105, 0.043796, 0.048921, 0.054506, 0.060586, 0.067202, 0.0744,
     0.08222, 0.0907, 0.099868, 0.109754, 0.120388, 0.131817, 0.144105, 0.157334, 0.171609, 0.187046,
     0.203765, 0.221873, 0.241451, 0.262539, 0.285129, 0.30916, 0.334529, 0.361101, 0.388727, 0.417257,
     0.446544, 0.476447, 0.50683, 0.537558, 0.568497, 1],
    [0.00062, 0.000456, 0.000337, 0.000256, 0.000203, 0.00017, 0.000149, 0.000137, 0.000133, 0.000136,
     0.000145, 0.000157, 0.000172, 0.000189, 0.000206, 0.000221, 0.000234, 0.000245, 0.000255, 0.000262,
     0.000269, 0.000274, 0.000279, 0.000284, 0.000289, 0.000294, 0.0003, 0.000307, 0.000316, 0.000327,
     0.00034, 0.000356, 0.000374, 0.000397, 0.000423, 0.000454, 0.000489, 0.00053, 0.000577, 0.000631,
     0.000692, 0.000762, 0.000841, 0.000929, 0.001028, 0.001137, 0.001259, 0.001392, 0.001537, 0.001692,
     0.001859, 0.002037, 0.002226, 0.002424, 0.002634, 0.002853, 0.003085, 0.003342, 0.003638, 0.00399,
     0.004414, 0.004923, 0.005529, 0.006244, 0.007078, 0.008045, 0.009165, 0.01046, 0.011955, 0.013674,
     0.015643, 0.017887, 0.020432, 0.023303, 0.026528, 0.030137, 0.034165, 0.038653, 0.043648, 0.049205,
     0.055385, 0.062254, 0.06988, 0.07832, 0.087611, 0.097754, 0.108704, 0.120371, 0.132638, 0.145395,
     0.158572, 0.172172, 0.186294, 0.201129, 0.21694, 0.234026, 0.252673, 0.273112, 0.295478, 0.319794,
     0.345975, 0.373856, 0.403221, 0.433833, 0.465447, 1]], name='CL13_I')


CL13_II = Probability([
    [0.00062, 0.000465, 0.000353, 0.000278, 0.000229, 0.0002, 0.000182, 0.000172, 0.000171, 0.000177,
     0.000187, 0.000202, 0.00022, 0.00024, 0.000261, 0.00028, 0.000298, 0.000315, 0.000331, 0.000346,
     0.000361, 0.000376, 0.000392, 0.000409, 0.000428, 0.000448, 0.000471, 0.000497, 0.000526, 0.000558,
     0.000595, 0.000635, 0.000681, 0.000732, 0.000788, 0.00085, 0.000919, 0.000995, 0.001078, 0.00117,
     0.00127, 0.00138, 0.0015, 0.001631, 0.001774, 0.001929, 0.002096, 0.002277, 0.002472, 0.002682,
     0.002908, 0.00315, 0.003409, 0.003686, 0.003982, 0.004297, 0.004636, 0.004999, 0.005389, 0.005807,
     0.006258, 0.006742, 0.007261, 0.007815, 0.008405, 0.009039, 0.009738, 0.010538, 0.011496, 0.012686,
     0.014192, 0.016106, 0.018517, 0.02151, 0.025151, 0.02949, 0.034545, 0.04031, 0.046747, 0.053801,
     0.061403, 0.069485, 0.077987, 0.086872, 0.09613, 0.105786, 0.1159, 0.126569, 0.137917, 0.150089,
     0.163239, 0.177519, 0.193067, 0.209999, 0.228394, 0.248299, 0.269718, 0.292621, 0.316951, 0.342628,
     0.369561, 0.397652, 0.426801, 0.456906, 0.487867, 1],
    [0.000455, 0.000324, 0.000236, 0.00018, 0.000149, 0.000131, 0.000119, 0.00011, 0.000105, 0.000103,
     0.000103, 0.000105, 0.000109, 0.000115, 0.000121, 0.000128, 0.000135, 0.000141, 0.000149, 0.000156,
     0.000163, 0.00017, 0.000178, 0.000185, 0.000192, 0.0002, 0.000208, 0.000216, 0.000225, 0.000235,
     0.000247, 0.000261, 0.000277, 0.000297, 0.000319, 0.000346, 0.000376, 0.000411, 0.00045, 0.000494,
     0.000542, 0.000595, 0.000653, 0.000715, 0.000783, 0.000857, 0.000935, 0.00102, 0.001112, 0.001212,
     0.001321, 0.001439, 0.001568, 0.001709, 0.001861, 0.002027, 0.002208, 0.002403, 0.002613, 0.00284,
     0.003088, 0.003366, 0.003684, 0.004055, 0.004495, 0.005016, 0.005626, 0.006326, 0.007115, 0.008,
     0.009007, 0.010185, 0.011606, 0.013353, 0.015508, 0.018134, 0.021268, 0.024916, 0.029062, 0.033674,
     0.038718, 0.04416, 0.049977, 0.056157, 0.062695, 0.069596, 0.076863, 0.084501, 0.092504, 0.100864,
     0.109567, 0.118605, 0.127985, 0.137743, 0.147962, 0.158777, 0.17038, 0.18302, 0.196986, 0.212604,
     0.230215, 0.250172, 0.272831, 0.298551, 0.327687, 1]], name='CL13_II')


CL13A = Probability([
    [0.000566, 0.000386, 0.000268, 0.000196, 0.000158, 0.000141, 0.000132, 0.000129, 0.000131, 0.000137,
     0.000146, 0.000157, 0.00017, 0.000184, 0.000197, 0.000208, 0.000219, 0.000227, 0.000235, 0.000241,
     0.000248, 0.000256, 0.000264, 0.000273, 0.000284, 0.000297, 0.000314, 0.000333, 0.000354, 0.000379,
     0.000407, 0.000438, 0.000472, 0.000509, 0.000549, 0.000592, 0.000639, 0.00069, 0.000746, 0.000808,
     0.000878, 0.000955, 0.001041, 0.001138, 0.001245, 0.001364, 0.001496, 0.001641, 0.001798, 0.001967,
     0.002148, 0.00234, 0.002544, 0.002759, 0.002985, 0.003221, 0.003469, 0.003731, 0.004014, 0.004323,
     0.00466, 0.005034, 0.005448, 0.005909, 0.006422, 0.006988, 0.00761, 0.008292, 0.009046, 0.009897,
     0.010888, 0.01208, 0.01355, 0.015387, 0.017686, 0.020539, 0.024017, 0.028162, 0.032978, 0.038437,
     0.044492, 0.051086, 0.058173, 0.065722, 0.073729, 0.082223, 0.091239, 0.1009, 0.111321, 0.122608,
     0.13487, 0.148212, 0.162742, 0.178566, 0.195793, 0.214499, 0.23465, 0.25618, 0.279025, 0.30312,
     0.328401, 0.354803, 0.382261, 0.41071, 0.440086, 1],
    [0.000453, 0.000289, 0.000184, 0.000124, 0.000095, 0.000084, 0.000078, 0.000074, 0.000072, 0.000072,
     0.000074, 0.000077, 0.00008, 0.000085, 0.00009, 0.000095, 0.0001, 0.000105, 0.00011, 0.000115,
     0.00012, 0.000125, 0.000129, 0.000134, 0.000139, 0.000144, 0.000149, 0.000154, 0.00016, 0.000167,
     0.000175, 0.000186, 0.000198, 0.000213, 0.000231, 0.000253, 0.000277, 0.000305, 0.000337, 0.000372,
     0.00041, 0.00045, 0.000494, 0.00054, 0.000589, 0.00064, 0.000693, 0.00075, 0.000811, 0.000877,
     0.00095, 0.001031, 0.00112,  0.001219, 0.001329, 0.00145, 0.001585, 0.001736, 0.001905, 0.002097,
     0.002315, 0.002561, 0.002836, 0.003137, 0.003468, 0.003835, 0.004254, 0.00474, 0.005302, 0.005943,
     0.00666, 0.00746, 0.008369, 0.009436, 0.01073, 0.012332, 0.014315, 0.016734, 0.019619, 0.022971,
     0.02677, 0.030989, 0.035598, 0.040576, 0.045915, 0.051616, 0.057646, 0.064084, 0.070942, 0.078241,
     0.086003, 0.094249, 0.103002, 0.112281, 0.122109, 0.13254, 0.143757, 0.155979, 0.169421, 0.184301,
     0.200836, 0.219242, 0.239737, 0.262537, 0.287859, 1]], name='CL13A')
     