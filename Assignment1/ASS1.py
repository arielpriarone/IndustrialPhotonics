# %%
import matplotlib.pyplot as plt
import matplotlib.colors as color
import numpy as np
import matplotlib
import tikzplotlib
import scipy.integrate as integrate
matplotlib.use('Qt5Agg')


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

def isNotebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    
if isNotebook(): # run widget only if in interactive mode
    get_ipython().run_line_magic('matplotlib', 'widget')


# %% compute the reflectivity 
n_si=3.8    # refraction coefficient of silicon
n_air=1     # refraction coefficient of air

R=((n_air-n_si)/(n_air+n_si))**2    # Normal incidence
R

theta=30/360*2*np.pi                # Angle of incidence, in radiants
R_tm=((n_air*np.sqrt(n_si**2-n_air**2*np.sin(theta)**2)-n_si**2*np.cos(theta))/
        (n_air*np.sqrt(n_si**2-n_air**2*np.sin(theta)**2)+n_si**2*np.cos(theta)))**2
R_te=((n_air*np.cos(theta)-np.sqrt(n_si**2-n_air**2*np.sin(theta)**2))/(n_air*np.cos(theta)+np.sqrt(n_si**2-n_air**2*np.sin(theta)**2)))**2
R_mean=np.mean([R_te,R_tm])
print(R_te)
print(R_tm)
print(R_mean)

# %% Sun tipical irradiance
AM15_lambda=np.array([300,      # wavelength
300.5,
301,
301.5,
302,
302.5,
303,
303.5,
304,
304.5,
305,
305.5,
306,
306.5,
307,
307.5,
308,
308.5,
309,
309.5,
310,
310.5,
311,
311.5,
312,
312.5,
313,
313.5,
314,
314.5,
315,
315.5,
316,
316.5,
317,
317.5,
318,
318.5,
319,
319.5,
320,
320.5,
321,
321.5,
322,
322.5,
323,
323.5,
324,
324.5,
325,
325.5,
326,
326.5,
327,
327.5,
328,
328.5,
329,
329.5,
330,
330.5,
331,
331.5,
332,
332.5,
333,
333.5,
334,
334.5,
335,
335.5,
336,
336.5,
337,
337.5,
338,
338.5,
339,
339.5,
340,
340.5,
341,
341.5,
342,
342.5,
343,
343.5,
344,
344.5,
345,
345.5,
346,
346.5,
347,
347.5,
348,
348.5,
349,
349.5,
350,
350.5,
351,
351.5,
352,
352.5,
353,
353.5,
354,
354.5,
355,
355.5,
356,
356.5,
357,
357.5,
358,
358.5,
359,
359.5,
360,
360.5,
361,
361.5,
362,
362.5,
363,
363.5,
364,
364.5,
365,
365.5,
366,
366.5,
367,
367.5,
368,
368.5,
369,
369.5,
370,
370.5,
371,
371.5,
372,
372.5,
373,
373.5,
374,
374.5,
375,
375.5,
376,
376.5,
377,
377.5,
378,
378.5,
379,
379.5,
380,
380.5,
381,
381.5,
382,
382.5,
383,
383.5,
384,
384.5,
385,
385.5,
386,
386.5,
387,
387.5,
388,
388.5,
389,
389.5,
390,
390.5,
391,
391.5,
392,
392.5,
393,
393.5,
394,
394.5,
395,
395.5,
396,
396.5,
397,
397.5,
398,
398.5,
399,
399.5,
400,
401,
402,
403,
404,
405,
406,
407,
408,
409,
410,
411,
412,
413,
414,
415,
416,
417,
418,
419,
420,
421,
422,
423,
424,
425,
426,
427,
428,
429,
430,
431,
432,
433,
434,
435,
436,
437,
438,
439,
440,
441,
442,
443,
444,
445,
446,
447,
448,
449,
450,
451,
452,
453,
454,
455,
456,
457,
458,
459,
460,
461,
462,
463,
464,
465,
466,
467,
468,
469,
470,
471,
472,
473,
474,
475,
476,
477,
478,
479,
480,
481,
482,
483,
484,
485,
486,
487,
488,
489,
490,
491,
492,
493,
494,
495,
496,
497,
498,
499,
500,
501,
502,
503,
504,
505,
506,
507,
508,
509,
510,
511,
512,
513,
514,
515,
516,
517,
518,
519,
520,
521,
522,
523,
524,
525,
526,
527,
528,
529,
530,
531,
532,
533,
534,
535,
536,
537,
538,
539,
540,
541,
542,
543,
544,
545,
546,
547,
548,
549,
550,
551,
552,
553,
554,
555,
556,
557,
558,
559,
560,
561,
562,
563,
564,
565,
566,
567,
568,
569,
570,
571,
572,
573,
574,
575,
576,
577,
578,
579,
580,
581,
582,
583,
584,
585,
586,
587,
588,
589,
590,
591,
592,
593,
594,
595,
596,
597,
598,
599,
600,
601,
602,
603,
604,
605,
606,
607,
608,
609,
610,
611,
612,
613,
614,
615,
616,
617,
618,
619,
620,
621,
622,
623,
624,
625,
626,
627,
628,
629,
630,
631,
632,
633,
634,
635,
636,
637,
638,
639,
640,
641,
642,
643,
644,
645,
646,
647,
648,
649,
650,
651,
652,
653,
654,
655,
656,
657,
658,
659,
660,
661,
662,
663,
664,
665,
666,
667,
668,
669,
670,
671,
672,
673,
674,
675,
676,
677,
678,
679,
680,
681,
682,
683,
684,
685,
686,
687,
688,
689,
690,
691,
692,
693,
694,
695,
696,
697,
698,
699,
700,
701,
702,
703,
704,
705,
706,
707,
708,
709,
710,
711,
712,
713,
714,
715,
716,
717,
718,
719,
720,
721,
722,
723,
724,
725,
726,
727,
728,
729,
730,
731,
732,
733,
734,
735,
736,
737,
738,
739,
740,
741,
742,
743,
744,
745,
746,
747,
748,
749,
750,
751,
752,
753,
754,
755,
756,
757,
758,
759,
760,
761,
762,
763,
764,
765,
766,
767,
768,
769,
770,
771,
772,
773,
774,
775,
776,
777,
778,
779,
780,
781,
782,
783,
784,
785,
786,
787,
788,
789,
790,
791,
792,
793,
794,
795,
796,
797,
798,
799,
800,
801,
802,
803,
804,
805,
806,
807,
808,
809,
810,
811,
812,
813,
814,
815,
816,
817,
818,
819,
820,
821,
822,
823,
824,
825,
826,
827,
828,
829,
830,
831,
832,
833,
834,
835,
836,
837,
838,
839,
840,
841,
842,
843,
844,
845,
846,
847,
848,
849,
850,
851,
852,
853,
854,
855,
856,
857,
858,
859,
860,
861,
862,
863,
864,
865,
866,
867,
868,
869,
870,
871,
872,
873,
874,
875,
876,
877,
878,
879,
880,
881,
882,
883,
884,
885,
886,
887,
888,
889,
890,
891,
892,
893,
894,
895,
896,
897,
898,
899,
900,
901,
902,
903,
904,
905,
906,
907,
908,
909,
910,
911,
912,
913,
914,
915,
916,
917,
918,
919,
920,
921,
922,
923,
924,
925,
926,
927,
928,
929,
930,
931,
932,
933,
934,
935,
936,
937,
938,
939,
940,
941,
942,
943,
944,
945,
946,
947,
948,
949,
950,
951,
952,
953,
954,
955,
956,
957,
958,
959,
960,
961,
962,
963,
964,
965,
966,
967,
968,
969,
970,
971,
972,
973,
974,
975,
976,
977,
978,
979,
980,
981,
982,
983,
984,
985,
986,
987,
988,
989,
990,
991,
992,
993,
994,
995,
996,
997,
998,
999,
1000,
1001,
1002,
1003,
1004,
1005,
1006,
1007,
1008,
1009,
1010,
1011,
1012,
1013,
1014,
1015,
1016,
1017,
1018,
1019,
1020,
1021,
1022,
1023,
1024,
1025,
1026,
1027,
1028,
1029,
1030,
1031,
1032,
1033,
1034,
1035,
1036,
1037,
1038,
1039,
1040,
1041,
1042,
1043,
1044,
1045,
1046,
1047,
1048,
1049,
1050,
1051,
1052,
1053,
1054,
1055,
1056,
1057,
1058,
1059,
1060,
1061,
1062,
1063,
1064,
1065,
1066,
1067,
1068,
1069,
1070,
1071,
1072,
1073,
1074,
1075,
1076,
1077,
1078,
1079,
1080,
1081,
1082,
1083,
1084,
1085,
1086,
1087,
1088,
1089,
1090,
1091,
1092,
1093,
1094,
1095,
1096,
1097,
1098,
1099,
1100,
1101,
1102,
1103,
1104,
1105,
1106,
1107,
1108,
1109,
1110,
1111,
1112,
1113,
1114,
1115,
1116,
1117,
1118,
1119,
1120,
1121,
1122,
1123,
1124,
1125,
1126,
1127,
1128,
1129,
1130,
1131,
1132,
1133,
1134,
1135,
1136,
1137,
1138,
1139,
1140,
1141,
1142,
1143,
1144,
1145,
1146,
1147,
1148,
1149,
1150,
1151,
1152,
1153,
1154,
1155,
1156,
1157,
1158,
1159,
1160,
1161,
1162,
1163,
1164,
1165,
1166,
1167,
1168,
1169,
1170,
1171,
1172,
1173,
1174,
1175,
1176,
1177,
1178,
1179,
1180,
1181,
1182,
1183,
1184,
1185,
1186,
1187,
1188,
1189,
1190,
1191,
1192,
1193,
1194,
1195,
1196,
1197,
1198,
1199,
1200,
1201,
1202,
1203,
1204,
1205,
1206,
1207,
1208,
1209,
1210,
1211,
1212,
1213,
1214,
1215,
1216,
1217,
1218,
1219,
1220,
1221,
1222,
1223,
1224,
1225,
1226,
1227,
1228,
1229,
1230,
1231,
1232,
1233,
1234,
1235,
1236,
1237,
1238,
1239,
1240,
1241,
1242,
1243,
1244,
1245,
1246,
1247,
1248,
1249,
1250,
1251,
1252,
1253,
1254,
1255,
1256,
1257,
1258,
1259,
1260,
1261,
1262,
1263,
1264,
1265,
1266,
1267,
1268,
1269,
1270,
1271,
1272,
1273,
1274,
1275,
1276,
1277,
1278,
1279,
1280,
1281,
1282,
1283,
1284,
1285,
1286,
1287,
1288,
1289,
1290,
1291,
1292,
1293,
1294,
1295,
1296,
1297,
1298,
1299,
1300,
1301,
1302,
1303,
1304,
1305,
1306,
1307,
1308,
1309,
1310,
1311,
1312,
1313,
1314,
1315,
1316,
1317,
1318,
1319,
1320,
1321,
1322,
1323,
1324,
1325,
1326,
1327,
1328,
1329,
1330,
1331,
1332,
1333,
1334,
1335,
1336,
1337,
1338,
1339,
1340,
1341,
1342,
1343,
1344,
1345,
1346,
1347,
1348,
1349,
1350,
1351,
1352,
1353,
1354,
1355,
1356,
1357,
1358,
1359,
1360,
1361,
1362,
1363,
1364,
1365,
1366,
1367,
1368,
1369,
1370,
1371,
1372,
1373,
1374,
1375,
1376,
1377,
1378,
1379,
1380,
1381,
1382,
1383,
1384,
1385,
1386,
1387,
1388,
1389,
1390,
1391,
1392,
1393,
1394,
1395,
1396,
1397,
1398,
1399,
1400,
1401,
1402,
1403,
1404,
1405,
1406,
1407,
1408,
1409,
1410,
1411,
1412,
1413,
1414,
1415,
1416,
1417,
1418,
1419,
1420,
1421,
1422,
1423,
1424,
1425,
1426,
1427,
1428,
1429,
1430,
1431,
1432,
1433,
1434,
1435,
1436,
1437,
1438,
1439,
1440,
1441,
1442,
1443,
1444,
1445,
1446,
1447,
1448,
1449,
1450,
1451,
1452,
1453,
1454,
1455,
1456,
1457,
1458,
1459,
1460,
1461,
1462,
1463,
1464,
1465,
1466,
1467,
1468,
1469,
1470,
1471,
1472,
1473,
1474,
1475,
1476,
1477,
1478,
1479,
1480,
1481,
1482,
1483,
1484,
1485,
1486,
1487,
1488,
1489,
1490,
1491,
1492,
1493,
1494,
1495,
1496,
1497,
1498,
1499,
1500,
1501,
1502,
1503,
1504,
1505,
1506,
1507,
1508,
1509,
1510,
1511,
1512,
1513,
1514,
1515,
1516,
1517,
1518,
1519,
1520,
1521,
1522,
1523,
1524,
1525,
1526,
1527,
1528,
1529,
1530,
1531,
1532,
1533,
1534,
1535,
1536,
1537,
1538,
1539,
1540,
1541,
1542,
1543,
1544,
1545,
1546,
1547,
1548,
1549,
1550,
1551,
1552,
1553,
1554,
1555,
1556,
1557,
1558,
1559,
1560,
1561,
1562,
1563,
1564,
1565,
1566,
1567,
1568,
1569,
1570,
1571,
1572,
1573,
1574,
1575,
1576,
1577,
1578,
1579,
1580,
1581,
1582,
1583,
1584,
1585,
1586,
1587,
1588,
1589,
1590,
1591,
1592,
1593,
1594,
1595,
1596,
1597,
1598,
1599,
1600,
1601,
1602,
1603,
1604,
1605,
1606,
1607,
1608,
1609,
1610,
1611,
1612,
1613,
1614,
1615,
1616,
1617,
1618,
1619,
1620,
1621,
1622,
1623,
1624,
1625,
1626,
1627,
1628,
1629,
1630,
1631,
1632,
1633,
1634,
1635,
1636,
1637,
1638,
1639,
1640,
1641,
1642,
1643,
1644,
1645,
1646,
1647,
1648,
1649,
1650,
1651,
1652,
1653,
1654,
1655,
1656,
1657,
1658,
1659,
1660,
1661,
1662,
1663,
1664,
1665,
1666,
1667,
1668,
1669,
1670,
1671,
1672,
1673,
1674,
1675,
1676,
1677,
1678,
1679,
1680,
1681,
1682,
1683,
1684,
1685,
1686,
1687,
1688,
1689,
1690,
1691,
1692,
1693,
1694,
1695,
1696,
1697,
1698,
1699,
1700,
1702,
1705,
1710,
1715,
1720,
1725,
1730,
1735,
1740,
1745,
1750,
1755,
1760,
1765,
1770,
1775,
1780,
1785,
1790,
1795,
1800,
1805,
1810,
1815,
1820,
1825,
1830,
1835,
1840,
1845,
1850,
1855,
1860,
1865,
1870,
1875,
1880,
1885,
1890,
1895,
1900,
1905,
1910,
1915,
1920,
1925,
1930,
1935,
1940,
1945,
1950,
1955,
1960,
1965,
1970,
1975,
1980,
1985,
1990,
1995,
2000,
2005,
2010,
2015,
2020,
2025,
2030,
2035,
2040,
2045,
2050,
2055,
2060,
2065,
2070,
2075,
2080,
2085,
2090,
2095,
2100,
2105,
2110,
2115,
2120,
2125,
2130,
2135,
2140,
2145,
2150,
2155,
2160,
2165,
2170,
2175,
2180,
2185,
2190,
2195,
2200,
2205,
2210,
2215,
2220,
2225,
2230,
2235,
2240,
2245,
2250,
2255,
2260,
2265,
2270,
2275,
2280,
2285,
2290,
2295,
2300,
2305,
2310,
2315,
2320,
2325,
2330,
2335,
2340,
2345,
2350,
2355,
2360,
2365,
2370,
2375,
2380,
2385,
2390,
2395,
2400,
2405,
2410,
2415,
2420,
2425,
2430,
2435,
2440,
2445,
2450,
2455,
2460,
2465,
2470,
2475,
2480,
2485,
2490,
2495,
2500])
AM15_irr=np.array([0.0010205,   # irradiance
0.001245,
0.00193,
0.0026914,
0.0029209,
0.004284,
0.0070945,
0.0089795,
0.0094701,
0.011953,
0.016463,
0.018719,
0.018577,
0.021108,
0.027849,
0.035635,
0.037837,
0.04143,
0.040534,
0.043306,
0.050939,
0.06554,
0.082922,
0.08408,
0.093376,
0.098984,
0.10733,
0.10757,
0.11969,
0.1306,
0.13625,
0.11838,
0.12348,
0.15036,
0.17158,
0.18245,
0.17594,
0.18591,
0.2047,
0.19589,
0.20527,
0.24525,
0.25024,
0.23843,
0.22203,
0.21709,
0.21226,
0.24861,
0.27537,
0.28321,
0.27894,
0.32436,
0.3812,
0.40722,
0.39806,
0.38465,
0.35116,
0.37164,
0.42235,
0.46878,
0.47139,
0.428,
0.40262,
0.41806,
0.43623,
0.43919,
0.42944,
0.40724,
0.41497,
0.44509,
0.46388,
0.45313,
0.41519,
0.38214,
0.3738,
0.40051,
0.43411,
0.45527,
0.46355,
0.47446,
0.5018,
0.50071,
0.47139,
0.46935,
0.48934,
0.50767,
0.51489,
0.48609,
0.41843,
0.40307,
0.45898,
0.48932,
0.47778,
0.48657,
0.49404,
0.47674,
0.47511,
0.48336,
0.46564,
0.47805,
0.52798,
0.56741,
0.55172,
0.53022,
0.51791,
0.48962,
0.5204,
0.57228,
0.60498,
0.61156,
0.6114,
0.59028,
0.55387,
0.51942,
0.45673,
0.46215,
0.43006,
0.39926,
0.46953,
0.56549,
0.59817,
0.56531,
0.52024,
0.50956,
0.5342,
0.5851,
0.60191,
0.58541,
0.60628,
0.60058,
0.62359,
0.68628,
0.73532,
0.73658,
0.72285,
0.70914,
0.66759,
0.6631,
0.69315,
0.74469,
0.75507,
0.68261,
0.69338,
0.72051,
0.67444,
0.64253,
0.61886,
0.55786,
0.5564,
0.55227,
0.5893,
0.65162,
0.6748,
0.6639,
0.71225,
0.79455,
0.85595,
0.83418,
0.74389,
0.66683,
0.70077,
0.75075,
0.76383,
0.68837,
0.58678,
0.50762,
0.45499,
0.44049,
0.50968,
0.61359,
0.67355,
0.64363,
0.621,
0.6457,
0.65147,
0.64204,
0.63582,
0.63136,
0.68543,
0.7597,
0.79699,
0.80371,
0.85138,
0.86344,
0.79493,
0.66257,
0.47975,
0.38152,
0.49567,
0.68385,
0.80772,
0.86038,
0.75655,
0.55017,
0.42619,
0.62945,
0.85249,
1.0069,
1.0693,
1.1021,
1.1141,
1.1603,
1.2061,
1.1613,
1.1801,
1.1511,
1.1227,
1.1026,
1.1514,
1.2299,
1.0485,
1.1738,
1.2478,
1.1971,
1.1842,
1.2258,
1.2624,
1.2312,
1.1777,
1.2258,
1.1232,
1.2757,
1.2583,
1.2184,
1.2117,
1.2488,
1.2135,
1.1724,
1.1839,
1.0963,
0.87462,
0.79394,
1.3207,
1.2288,
1.1352,
1.2452,
1.3659,
1.3943,
1.2238,
1.1775,
1.3499,
1.3313,
1.425,
1.4453,
1.4084,
1.4619,
1.3108,
1.4903,
1.5081,
1.5045,
1.5595,
1.6173,
1.5482,
1.4297,
1.5335,
1.5224,
1.5724,
1.5854,
1.5514,
1.5391,
1.5291,
1.5827,
1.5975,
1.6031,
1.5544,
1.535,
1.5673,
1.4973,
1.5619,
1.5682,
1.5077,
1.5331,
1.6126,
1.5499,
1.5671,
1.6185,
1.5631,
1.5724,
1.623,
1.5916,
1.6181,
1.6177,
1.6236,
1.6038,
1.5734,
1.5683,
1.2716,
1.4241,
1.5413,
1.4519,
1.6224,
1.5595,
1.4869,
1.5903,
1.5525,
1.6485,
1.5676,
1.5944,
1.5509,
1.5507,
1.5451,
1.4978,
1.4966,
1.5653,
1.4587,
1.5635,
1.6264,
1.556,
1.5165,
1.5893,
1.5481,
1.5769,
1.6186,
1.5206,
1.4885,
1.5314,
1.5455,
1.2594,
1.4403,
1.3957,
1.5236,
1.5346,
1.569,
1.4789,
1.5905,
1.5781,
1.5341,
1.3417,
1.5357,
1.6071,
1.5446,
1.6292,
1.5998,
1.4286,
1.5302,
1.5535,
1.6199,
1.4989,
1.5738,
1.5352,
1.4825,
1.4251,
1.5511,
1.5256,
1.5792,
1.5435,
1.5291,
1.549,
1.5049,
1.552,
1.5399,
1.5382,
1.5697,
1.525,
1.5549,
1.5634,
1.5366,
1.4988,
1.531,
1.4483,
1.474,
1.5595,
1.4847,
1.5408,
1.5106,
1.5201,
1.4374,
1.532,
1.518,
1.4807,
1.4816,
1.4331,
1.5134,
1.5198,
1.5119,
1.4777,
1.4654,
1.5023,
1.456,
1.477,
1.502,
1.5089,
1.532,
1.5479,
1.5448,
1.5324,
1.4953,
1.5281,
1.4934,
1.2894,
1.3709,
1.4662,
1.4354,
1.4561,
1.4491,
1.4308,
1.4745,
1.4788,
1.4607,
1.4606,
1.4753,
1.4579,
1.436,
1.4664,
1.4921,
1.4895,
1.4822,
1.4911,
1.4862,
1.4749,
1.4686,
1.4611,
1.4831,
1.4621,
1.4176,
1.4697,
1.431,
1.4128,
1.4664,
1.4733,
1.4739,
1.4802,
1.4269,
1.4165,
1.4118,
1.4026,
1.4012,
1.4417,
1.3631,
1.4114,
1.3924,
1.4161,
1.3638,
1.4508,
1.4284,
1.4458,
1.4128,
1.461,
1.4707,
1.4646,
1.434,
1.4348,
1.4376,
1.4525,
1.4462,
1.4567,
1.415,
1.4086,
1.3952,
1.3519,
1.3594,
1.4447,
1.3871,
1.4311,
1.4153,
1.3499,
1.1851,
1.2393,
1.3855,
1.3905,
1.3992,
1.3933,
1.3819,
1.3844,
1.3967,
1.4214,
1.4203,
1.4102,
1.415,
1.4394,
1.4196,
1.4169,
1.3972,
1.4094,
1.4074,
1.3958,
1.412,
1.3991,
1.4066,
1.3947,
1.3969,
1.3915,
1.3981,
1.383,
1.3739,
1.3748,
1.3438,
0.96824,
1.1206,
1.1278,
1.1821,
1.2333,
1.2689,
1.2609,
1.2464,
1.2714,
1.2684,
1.3403,
1.3192,
1.2918,
1.2823,
1.2659,
1.2674,
1.2747,
1.3078,
1.3214,
1.3144,
1.309,
1.3048,
1.3095,
1.3175,
1.3155,
1.3071,
1.2918,
1.3029,
1.2587,
1.2716,
1.1071,
1.0296,
0.92318,
0.9855,
1.0861,
1.2407,
1.1444,
1.0555,
1.038,
1.0813,
1.085,
1.04,
1.0466,
1.1285,
1.0703,
1.1534,
1.1962,
1.2357,
1.2178,
1.2059,
1.2039,
1.2269,
1.1905,
1.2195,
1.2148,
1.2153,
1.2405,
1.2503,
1.2497,
1.247,
1.2477,
1.2401,
1.2357,
1.2341,
1.2286,
1.233,
1.2266,
1.242,
1.2383,
1.2232,
1.2221,
1.2295,
1.1945,
0.26604,
0.15396,
0.68766,
0.37952,
0.53878,
0.68601,
0.81461,
0.97417,
1.1138,
1.1278,
1.1608,
1.1686,
1.1778,
1.1771,
1.1771,
1.1771,
1.1798,
1.1727,
1.1713,
1.1765,
1.1636,
1.1607,
1.1662,
1.1614,
1.1536,
1.1586,
1.1592,
1.145,
1.1305,
1.1257,
1.091,
1.1058,
1.0953,
1.0875,
1.0972,
1.0932,
1.0742,
1.0913,
1.1121,
1.0905,
1.0725,
1.0843,
1.0856,
1.0657,
1.0782,
1.0545,
1.0974,
1.0859,
1.0821,
1.0548,
1.0559,
1.0533,
1.0268,
1.0086,
0.90356,
0.89523,
0.83216,
0.85183,
0.82259,
0.90519,
0.86188,
0.99764,
0.95157,
0.67271,
0.93506,
0.96935,
0.93381,
0.98465,
0.84979,
0.9293,
0.91601,
0.92392,
0.89426,
0.9565,
0.93412,
1.0032,
0.97234,
1.0092,
0.99901,
1.0013,
1.0157,
1.0101,
0.99703,
1.0053,
0.98631,
1.0165,
1.0187,
0.9917,
0.99217,
0.98596,
0.89372,
0.97493,
0.96927,
0.96486,
0.85112,
0.913,
0.97317,
0.99166,
0.99196,
0.99171,
0.98816,
0.98679,
0.99449,
1.0005,
0.97916,
0.96324,
0.849,
0.91546,
0.9592,
0.94956,
0.96755,
0.95387,
0.96686,
0.95721,
0.94042,
0.92687,
0.95277,
0.95615,
0.95237,
0.93656,
0.93957,
0.90861,
0.93245,
0.92927,
0.93305,
0.94423,
0.90752,
0.91062,
0.92228,
0.93455,
0.92393,
0.92584,
0.90881,
0.87327,
0.8513,
0.81357,
0.76253,
0.66566,
0.7178,
0.54871,
0.7426,
0.59933,
0.66791,
0.68889,
0.84457,
0.81709,
0.77558,
0.63854,
0.65217,
0.70431,
0.62467,
0.66808,
0.68893,
0.62834,
0.62649,
0.67836,
0.57646,
0.73017,
0.59271,
0.73877,
0.74414,
0.78049,
0.70026,
0.74504,
0.7215,
0.7111,
0.70331,
0.78742,
0.58968,
0.55127,
0.4321,
0.40921,
0.30086,
0.24841,
0.1438,
0.25084,
0.16142,
0.16338,
0.20058,
0.39887,
0.47181,
0.37195,
0.40532,
0.27834,
0.28579,
0.36821,
0.19461,
0.37112,
0.27423,
0.49396,
0.14726,
0.48378,
0.26891,
0.34362,
0.42411,
0.34117,
0.32821,
0.27067,
0.46101,
0.37385,
0.42066,
0.4612,
0.44174,
0.50503,
0.4586,
0.50374,
0.50275,
0.5024,
0.6521,
0.68622,
0.63461,
0.71397,
0.68765,
0.60648,
0.57529,
0.58987,
0.57191,
0.63864,
0.61509,
0.63815,
0.60468,
0.71338,
0.69218,
0.66865,
0.73732,
0.68817,
0.75083,
0.73928,
0.73462,
0.74906,
0.73227,
0.75358,
0.75102,
0.73728,
0.7541,
0.75176,
0.74884,
0.73971,
0.73887,
0.73857,
0.73532,
0.74442,
0.72805,
0.73442,
0.72336,
0.68174,
0.71252,
0.72753,
0.72685,
0.71972,
0.71914,
0.72278,
0.71877,
0.71761,
0.72068,
0.70817,
0.71129,
0.70337,
0.71422,
0.68878,
0.69896,
0.70175,
0.6897,
0.69508,
0.69058,
0.69753,
0.69636,
0.69305,
0.69385,
0.68628,
0.69055,
0.68736,
0.68787,
0.67613,
0.68015,
0.68234,
0.68202,
0.67497,
0.67172,
0.67636,
0.6717,
0.67176,
0.672,
0.66525,
0.66833,
0.66452,
0.64714,
0.65694,
0.66274,
0.65896,
0.65463,
0.65521,
0.65118,
0.64919,
0.64646,
0.64847,
0.64641,
0.64482,
0.63818,
0.61875,
0.63585,
0.62121,
0.63266,
0.62239,
0.63196,
0.62913,
0.61713,
0.62032,
0.61944,
0.58626,
0.60469,
0.61661,
0.61536,
0.60363,
0.62158,
0.59252,
0.61471,
0.60434,
0.60321,
0.60474,
0.59722,
0.58083,
0.5894,
0.59814,
0.57852,
0.5933,
0.5541,
0.56697,
0.59317,
0.57919,
0.55573,
0.58835,
0.58124,
0.51058,
0.53965,
0.52067,
0.50323,
0.57852,
0.50291,
0.50772,
0.48577,
0.49696,
0.46883,
0.46637,
0.46765,
0.50644,
0.39792,
0.48304,
0.41565,
0.41278,
0.47899,
0.33154,
0.41357,
0.2685,
0.29985,
0.24987,
0.20136,
0.079618,
0.21753,
0.11317,
0.14189,
0.18586,
0.081686,
0.12817,
0.1087,
0.14428,
0.051589,
0.15725,
0.099224,
0.10591,
0.070574,
0.2956,
0.23411,
0.15331,
0.04174,
0.015462,
0.12876,
0.28785,
0.20329,
0.2985,
0.25599,
0.19337,
0.22479,
0.31183,
0.11326,
0.14604,
0.15764,
0.059176,
0.27113,
0.21854,
0.12164,
0.2034,
0.24762,
0.23812,
0.14248,
0.31316,
0.2809,
0.31458,
0.31171,
0.33693,
0.28648,
0.34753,
0.35002,
0.46857,
0.40188,
0.3886,
0.37494,
0.40996,
0.41954,
0.4231,
0.45873,
0.44831,
0.45483,
0.45642,
0.33692,
0.4524,
0.47679,
0.47235,
0.36,
0.48371,
0.44069,
0.45514,
0.32318,
0.4387,
0.41985,
0.40741,
0.47715,
0.45575,
0.33504,
0.41569,
0.46239,
0.4466,
0.47336,
0.45434,
0.4689,
0.44696,
0.43131,
0.47715,
0.43392,
0.36489,
0.44825,
0.43708,
0.43717,
0.43409,
0.36247,
0.43692,
0.48086,
0.42986,
0.43346,
0.41428,
0.45336,
0.42232,
0.42489,
0.46956,
0.43407,
0.4278,
0.4664,
0.45528,
0.45934,
0.44663,
0.45805,
0.46531,
0.45139,
0.44406,
0.44808,
0.46236,
0.46819,
0.43304,
0.46658,
0.46721,
0.46003,
0.47203,
0.46633,
0.45397,
0.47016,
0.46504,
0.46908,
0.46339,
0.46797,
0.46272,
0.46077,
0.46197,
0.46247,
0.45754,
0.45528,
0.45655,
0.45945,
0.45746,
0.4586,
0.45966,
0.45705,
0.45258,
0.45097,
0.44773,
0.44363,
0.4507,
0.44023,
0.43532,
0.44496,
0.42725,
0.4311,
0.41146,
0.39567,
0.40019,
0.37148,
0.3957,
0.38527,
0.38822,
0.37051,
0.24652,
0.38744,
0.40825,
0.40879,
0.40625,
0.40614,
0.41233,
0.41693,
0.42001,
0.42763,
0.42456,
0.42204,
0.41335,
0.37305,
0.40733,
0.42078,
0.42399,
0.42714,
0.42213,
0.41989,
0.40936,
0.41285,
0.41786,
0.39618,
0.41257,
0.40421,
0.40514,
0.38957,
0.3713,
0.39183,
0.40852,
0.35312,
0.36228,
0.39181,
0.34621,
0.30062,
0.38382,
0.38453,
0.30594,
0.34696,
0.38413,
0.30114,
0.33366,
0.33337,
0.31352,
0.28833,
0.28581,
0.32419,
0.31217,
0.33328,
0.26855,
0.25872,
0.29866,
0.30217,
0.23279,
0.26249,
0.32224,
0.28051,
0.26625,
0.2345,
0.17759,
0.22923,
0.1448,
0.14579,
0.20304,
0.16925,
0.23117,
0.18348,
0.16454,
0.17804,
0.17681,
0.16831,
0.17039,
0.17798,
0.12711,
0.075645,
0.10904,
0.058186,
0.060119,
0.0047451,
0.016159,
0.016025,
0.0046298,
0.0015164,
0.000096096,
0.00029009,
0.0000036034,
0.00004807,
0.000071786,
0.0000041948,
0.00000073439,
0.0000021404,
0.0000000048133,
0.000000000018076,
0.0000031563,
0.0000013589,
0.0000000000090764,
0.000012791,
0.0000049764,
0.0000000000001481,
0.00000051667,
0.000000292,
0.000000019731,
0.0000027498,
0.000044401,
0.00017917,
0.00032332,
0.00025748,
0.0001227,
0.0011089,
0.000052164,
0.000081587,
0.0000023716,
0.0000025672,
0.000000044017,
0.00000061689,
0.0000020899,
0.0000025215,
0.00019896,
0.0000040262,
0.00058098,
0.00049328,
0.00034384,
0.000023782,
0.00011586,
0.000075526,
0.00000067136,
0.0000000063215,
0.000049057,
0.0012704,
0.00081226,
0.0000000032466,
0.000000010528,
0.0018353,
0.00238,
0.00073892,
0.00000036444,
0.0020448,
0.00017457,
0.0016493,
0.00061919,
0.00046653,
0.0021142,
0.0026396,
0.023353,
0.00036378,
0.00018366,
0.035565,
0.011759,
0.013559,
0.0021442,
0.0082718,
0.0091637,
0.046314,
0.0092198,
0.016975,
0.02585,
0.027792,
0.049546,
0.0045588,
0.03802,
0.061601,
0.050156,
0.0025194,
0.035834,
0.020962,
0.021416,
0.038351,
0.02988,
0.013263,
0.051039,
0.039601,
0.0318,
0.036317,
0.045063,
0.061791,
0.049751,
0.023095,
0.036215,
0.11569,
0.10213,
0.027412,
0.011271,
0.062361,
0.081978,
0.13759,
0.06615,
0.088509,
0.117,
0.13643,
0.16307,
0.085421,
0.090276,
0.1306,
0.043225,
0.15184,
0.093383,
0.065197,
0.036054,
0.076942,
0.094845,
0.049678,
0.017848,
0.046771,
0.070198,
0.097339,
0.18463,
0.068778,
0.069736,
0.06348,
0.12001,
0.060637,
0.11529,
0.05849,
0.14859,
0.13747,
0.12503,
0.1234,
0.060629,
0.09418,
0.18973,
0.17478,
0.19778,
0.16441,
0.18157,
0.20367,
0.18253,
0.16852,
0.2285,
0.18968,
0.21759,
0.25061,
0.26552,
0.23356,
0.18493,
0.16029,
0.18402,
0.25773,
0.25514,
0.24302,
0.1869,
0.27052,
0.26474,
0.26068,
0.24239,
0.22571,
0.26573,
0.25683,
0.24929,
0.25211,
0.24437,
0.2645,
0.27505,
0.26378,
0.28004,
0.27539,
0.25884,
0.26745,
0.2622,
0.27928,
0.27244,
0.25522,
0.26973,
0.27839,
0.27714,
0.26892,
0.26686,
0.27464,
0.27336,
0.27202,
0.27295,
0.26491,
0.26904,
0.26927,
0.27208,
0.2721,
0.27705,
0.27481,
0.27309,
0.26675,
0.27342,
0.2699,
0.27058,
0.27182,
0.27132,
0.26474,
0.26759,
0.2631,
0.27062,
0.26848,
0.26808,
0.26568,
0.27002,
0.26756,
0.26667,
0.26264,
0.26728,
0.26245,
0.26308,
0.25722,
0.25452,
0.24175,
0.23507,
0.23775,
0.23407,
0.24145,
0.23974,
0.24678,
0.21602,
0.23516,
0.23672,
0.24464,
0.2487,
0.24195,
0.24755,
0.24904,
0.25874,
0.25569,
0.25303,
0.25107,
0.23233,
0.24179,
0.24197,
0.25225,
0.25833,
0.25624,
0.25823,
0.24452,
0.24692,
0.25421,
0.24202,
0.2381,
0.22323,
0.22413,
0.22397,
0.22842,
0.23683,
0.2414,
0.23296,
0.2299,
0.22727,
0.2176,
0.2268,
0.23076,
0.23719,
0.23838,
0.24104,
0.2305,
0.23465,
0.24352,
0.241,
0.23449,
0.2343,
0.23754,
0.24246,
0.24269,
0.23782,
0.23971,
0.24078,
0.24126,
0.24137,
0.23651,
0.23806,
0.23821,
0.23267,
0.23282,
0.23367,
0.23539,
0.227,
0.22007,
0.22026,
0.21511,
0.2196,
0.22082,
0.21535,
0.22355,
0.21822,
0.21749,
0.22768,
0.21655,
0.21867,
0.22526,
0.20855,
0.22373,
0.22277,
0.21583,
0.22231,
0.22101,
0.22223,
0.22487,
0.2212,
0.22332,
0.22384,
0.21908,
0.22235,
0.22098,
0.21178,
0.17884,
0.21068,
0.21459,
0.21516,
0.22168,
0.21879,
0.21147,
0.21629,
0.21575,
0.2136,
0.21145,
0.21229,
0.20915,
0.21303,
0.20558,
0.19447,
0.20366,
0.20906,
0.19797,
0.21321,
0.21026,
0.20484,
0.21013,
0.20718,
0.20523,
0.19303,
0.20708,
0.21134,
0.20477,
0.20968,
0.20922,
0.18107,
0.20739,
0.20551,
0.19975,
0.20396,
0.19778,
0.1879,
0.18965,
0.18698,
0.17808,
0.17407,
0.16154,
0.16818,
0.15481,
0.16566,
0.15301,
0.15998,
0.13284,
0.14172,
0.11484,
0.1005,
0.076981,
0.088904,
0.046931,
0.031828,
0.014815,
0.0096911,
0.0032816,
0.00098755,
0.0012744,
0.0000052041,
0.000006419,
0.000000062703,
0.0000062658,
0.0000029993,
0.00000028396,
0.000011151,
0.000016982,
0.00000000026662,
0.0000000004513,
0.000077505,
0.00004389,
0.00022333,
0.00012947,
0.00000086221,
0.00000056667,
0.000023045,
0.000019947,
0.00045069,
0.00093615,
0.00055242,
0.0035935,
0.0032821,
0.010863,
0.016727,
0.010036,
0.021906,
0.028563,
0.048847,
0.067857,
0.075512,
0.083063,
0.085613,
0.08119,
0.038156,
0.015001,
0.039748,
0.026648,
0.044981,
0.07401,
0.084856,
0.096386,
0.089781,
0.091074,
0.067927,
0.054906,
0.069193,
0.061875,
0.065676,
0.077443,
0.086812,
0.085102,
0.0891,
0.089747,
0.086133,
0.093153,
0.089654,
0.091673,
0.087588,
0.088632,
0.089774,
0.090044,
0.090767,
0.089486,
0.084639,
0.08484,
0.08417,
0.07631,
0.081996,
0.080448,
0.081808,
0.07455,
0.079068,
0.078992,
0.071202,
0.07401,
0.079315,
0.076273,
0.07773,
0.075453,
0.075773,
0.074299,
0.073118,
0.070838,
0.071937,
0.06769,
0.066929,
0.068137,
0.064867,
0.064021,
0.066288,
0.06308,
0.06322,
0.061265,
0.058824,
0.059171,
0.06387,
0.058141,
0.052031,
0.056215,
0.056824,
0.057967,
0.045836,
0.0514,
0.041536,
0.047473,
0.050237,
0.049409,
0.030817,
0.044147,
0.042552,
0.030826,
0.037109,
0.040594,
0.04415,
0.033599,
0.033813,
0.0273,
0.02659,
0.033078,
0.045099,
0.014878,
0.043249,
0.020798,
0.013611,
0.024853,
0.033363,
0.024148,
0.016727,
0.016455,
0.0080395,
0.0056102,
0.0035113,
0.0028772,
0.0070642])

fig, axs=plt.subplots()
fig.tight_layout()
axs.semilogx(AM15_lambda,AM15_irr)
axs.set_xlabel('$\lambda$ [nm]')
axs.set_ylabel('Irradiance [${W}/{(m^2\cdot nm)}$]')
axs.grid(True,'both')
tikzplotlib.save('Assignment1/irradiance.tex',axis_width='0.9\\textwidth',axis_height ='6cm')

# %% sLARC design
lambda0=495 # design wavelength [nm]
n_ideal=np.sqrt(n_air*n_si) # ideal refractive index at lambda0
d_ideal=lambda0/4/n_ideal # ideal thickness
n_TiO2= 2.7193              # at 495 nm
d_TiO2= lambda0/4/n_TiO2    # optimized for lambda0
n_Si3N4 = 2.0647            # at 495 nm
d_Si3N4=lambda0/4/n_Si3N4   # optimized for lambda0
n_Al2O3 = 1.7747            # at 495 nm
d_Al2O3=lambda0/4/n_Al2O3   # optimized for lambda0
n_SiO2  = 1.4626            # at 495 nm
d_SiO2=lambda0/4/n_SiO2     # optimized for lambda0

lambdas=np.logspace(np.log10(300),np.log10(2500),200) # 200 points in the region of interest
lambdas=AM15_lambda # use the AM15 vector already present

# %% check reflectivity - no cromatic dispersion
fig, axs=plt.subplots()
fig.tight_layout()
for n_ar, d in zip([n_ideal,n_Al2O3,n_Si3N4,n_SiO2,n_TiO2], [d_ideal,d_Al2O3,d_Si3N4,d_SiO2,d_TiO2]):
    print(n_ar,d)
    Z0=120*np.pi                # free space inpedance
    Z_inf_ar=Z0/n_ar            # impedence of ARC
    Z_inf_Si=Z0/n_si            # impedence of bulk silicon
    Gbminus=(Z_inf_Si-Z_inf_ar)/(Z_inf_Si+Z_inf_ar) # reflection coefficient
    K0=2*np.pi/lambdas
    k=K0*n_ar
    Gaplus=(Gbminus*np.exp(-2j*k*d))
    Za=Z_inf_ar*(1+Gaplus)/(1-Gaplus)
    Gaminus=(Za-Z0)/(Za+Z0)
    R_vect=np.abs(Gaminus)**2*100
    axs.plot(lambdas,R_vect)


axs.set_xlabel('$\lambda$ [nm]')
axs.set_ylabel('Reflectivity [%]')
axs.grid(True,'both')
axs.minorticks_on()
axs.legend(['ideal $n=1.9493$', '$Al_2O_3 \, n=1.7747$', '$Si_3N_4 \, n=2.0647$', '$SiO_2 \, n=1.4626$', '$TiO_2 \, n=2.7193$'])
tikzplotlib_fix_ncols(fig)
tikzplotlib.save('Assignment1/SLARC.tex',axis_width='0.9\\textwidth',axis_height ='7cm')


# %% compute the chromatic dispersion

n_ideal_v=n_ideal*lambdas/lambdas # ideal refractive index at lambda0
lambdas_um=lambdas/1000
n_TiO2_v = (4.99+1/96.6*lambdas_um**(-1.1)+1/4.6*lambdas_um**(-1.95))**.5             # at every lambda 
n_Si3N4_v = (1+3.0249/(1-(0.1353406/lambdas_um)**2)+40314/(1-(1239.842/lambdas_um)**2))**.5            # at every lambda
n_Al2O3_v = (1+1.4313493/(1-(0.0726631/lambdas_um)**2)+0.65054713/(1-(0.1193242/lambdas_um)**2)+5.3414021/(1-(18.028251/lambdas_um)**2))**.5            # at every lambda
n_SiO2_v  = (1+0.6961663/(1-(0.0684043/lambdas_um)**2)+0.4079426/(1-(0.1162414/lambdas_um)**2)+0.8974794/(1-(9.896161/lambdas_um)**2))**.5            # at every lambda

fig, axs=plt.subplots()
fig.tight_layout()
for n_ar in [n_ideal_v,n_Al2O3_v,n_Si3N4_v,n_SiO2_v,n_TiO2_v]:
    axs.plot(lambdas,n_ar)
axs.set_xlabel('$\lambda$ [nm]')
axs.set_ylabel('$n$')
axs.grid(True,'both')
axs.minorticks_on()
axs.legend(['ideal', '$Al_2O_3$', '$Si_3N_4$', '$SiO_2$', '$TiO_2$'])
tikzplotlib_fix_ncols(fig)
tikzplotlib.save('Assignment1/n_dispersion.tex',axis_width='0.9\\textwidth',axis_height ='5cm')

# %% check reflectivity - with dispersion
fig, axs=plt.subplots()
fig.tight_layout()
for n_ar, d in zip([n_ideal_v,n_Al2O3_v,n_Si3N4_v,n_SiO2_v,n_TiO2_v], [d_ideal,d_Al2O3,d_Si3N4,d_SiO2,d_TiO2]):
    Z0=120*np.pi                # free space inpedance
    Z_inf_ar=Z0/n_ar            # impedence of ARC
    Z_inf_Si=Z0/n_si            # impedence of bulk silicon
    Gbminus=(Z_inf_Si-Z_inf_ar)/(Z_inf_Si+Z_inf_ar) # reflection coefficient
    k=K0*n_ar
    Gaplus=(Gbminus*np.exp(-2j*k*d))
    Za=Z_inf_ar*(1+Gaplus)/(1-Gaplus)
    Gaminus=(Za-Z0)/(Za+Z0)
    R_vect=np.abs(Gaminus)**2*100
    if n_ar is n_Si3N4_v:
        R_SLARC=R_vect #save the best design
    axs.plot(lambdas,R_vect)


axs.set_xlabel('$\lambda$ [nm]')
axs.set_ylabel('Reflectivity [%]')
axs.grid(True,'both')
axs.minorticks_on()
axs.legend(['ideal', '$Al_2O_3$', '$Si_3N_4$', '$SiO_2$', '$TiO_2$'])
tikzplotlib_fix_ncols(fig)
tikzplotlib.save('Assignment1/SLARC_n_vary.tex',axis_width='0.9\\textwidth',axis_height ='7cm')

# %% DL ARC - some n
n_dict={"$Na_3AlF_6$": 1.35,
        "$MgF_2$": 1.38,
        "$SiO_2$": 1.46,
        "$Al_2O_3$": 1.629,
        "$CeF_3$": 1.63,
        "$PbF_2$": 1.73,
        "$Ta_2O_5$": 2.126,
        "$ZrO_2$": 2.20,
        "$ZnS$": 2.32,
        "$TiO_2$": 2.40,
        "$Bi_2O_3$": 2.45,
        "$Ge$": 4.2,
        "$Te$": 4.60}


# %% DLARC - SILICA/TITANIA
Z_inf_SiO2=Z0/n_SiO2
Z_inf_TiO2=Z0/n_TiO2
Gcminus=(Z_inf_Si-Z_inf_TiO2)/(Z_inf_Si+Z_inf_TiO2)
k=K0*n_dict['$TiO_2$']
Gbplus=Gcminus*np.exp(-2j*k*d_TiO2)
Zb=Z_inf_TiO2*(1+Gbplus)/(1-Gbplus)
Gbminus=(Zb-Z_inf_SiO2)/(Zb+Z_inf_SiO2)
k=K0*n_dict['$SiO_2$']
Gaplus=Gbminus*np.exp(-2j*k*d_SiO2)
Za=Z_inf_SiO2*(1+Gaplus)/(1-Gaplus)
Gaminus=(Za-Z0)/(Za+Z0)
R_DLARC_SiO2TiO2=np.abs(Gaminus)**2*100

# %% DL ARC - my design
legend=[]
val=[]
for keys_e, values_e in n_dict.items():
    for keys_i, values_i in n_dict.items():
        val.append(values_e/values_i)
        legend.append(keys_e + "//" + keys_i)

fig, axs=plt.subplots()
fig.tight_layout()
axs.plot(legend,val,'x')
axs.hlines([(n_air/n_si)**0.5],legend[0],legend[-1],'red')
axs.set_xlabel('Combination of materials')
axs.set_ylabel('$n_1/n_2$')
axs.grid(True,'both')
axs.legend(['available ratios', 'reference'])
tikzplotlib_fix_ncols(fig)
tikzplotlib.save('Assignment1/DLARC_ratios.tex',axis_width='0.9\\textwidth',axis_height ='7cm')

# %% DLARC - my design
d_Ge=lambda0/4/n_dict['$Ge$']
d_Ta=lambda0/4/n_dict['$Ta_2O_5$']
Z_inf_Ge=Z0/n_dict['$Ge$']
Z_inf_Ta=Z0/n_dict['$Ta_2O_5$']
Gcminus=(Z_inf_Si-Z_inf_Ge)/(Z_inf_Si+Z_inf_Ge)
k=K0*n_dict['$Ge$']
Gbplus=Gcminus*np.exp(-2j*k*d_Ge)
Zb=Z_inf_Ge*(1+Gbplus)/(1-Gbplus)
Gbminus=(Zb-Z_inf_Ta)/(Zb+Z_inf_Ta)
k=K0*n_dict['$Ta_2O_5$']
Gaplus=Gbminus*np.exp(-2j*k*d_Ta)
Za=Z_inf_Ta*(1+Gaplus)/(1-Gaplus)
Gaminus=(Za-Z0)/(Za+Z0)
R_DLARC_Ta2O5Ge=np.abs(Gaminus)**2*100




# %% check reflectivity - SLARC / DLARCs
fig, axs=plt.subplots()
fig.tight_layout()
axs.semilogy(lambdas,R_DLARC_SiO2TiO2)
axs.semilogy(lambdas,R_DLARC_Ta2O5Ge)
axs.semilogy(lambdas,R_SLARC)

axs.set_xlabel('$\lambda$ [nm]')
axs.set_ylabel('Reflectivity [%]')
axs.grid(True,'both')
axs.minorticks_on()
axs.legend(['DLARC $TiO2 \,/\, SiO2$','DLARC $Ta_2O_5 \,/\, Ge$','SLARC $Si_3N_4$'])
tikzplotlib_fix_ncols(fig)
tikzplotlib.save('Assignment1/DLARC.tex',axis_width='0.9\\textwidth',axis_height ='7cm')

# %% check effective reflectivity
indx=0
R_eff=[]; R_min=[]
for R in [R_DLARC_SiO2TiO2, R_DLARC_Ta2O5Ge, R_SLARC]:
    NUM=0; DEN=0
    for _ in lambdas:
        if indx is (len(lambdas)-2):
            break
        NUM+=R[indx]*AM15_irr[indx]*(lambdas[indx+1]-lambdas[indx])
        DEN+=AM15_irr[indx]*(lambdas[indx+1]-lambdas[indx])
    R_eff.append(NUM/DEN)
    R_min.append(np.min(R))
print(R_eff);print(R_min)

# %% check effective reflectivity in range of PV
indx=0
R_eff=[]
for R in [R_DLARC_SiO2TiO2, R_DLARC_Ta2O5Ge, R_SLARC]:
    NUM=0
    DEN=0
    for lam in lambdas:
        if indx is (len(lambdas)-2):
            break
        if 400 <= lam <= 1100:
           NUM+=R[indx]*AM15_irr[indx]*(lambdas[indx+1]-lambdas[indx])
           DEN+=AM15_irr[indx]*(lambdas[indx+1]-lambdas[indx])
    R_eff.append(NUM/DEN); print(R_eff)


# %% DLARC - tolerance effect
Z_inf_SiO2=Z0/n_dict['$SiO_2$']
Z_inf_TiO2=Z0/n_dict['$TiO_2$']
Gcminus=(Z_inf_Si-Z_inf_TiO2)/(Z_inf_Si+Z_inf_TiO2)
k=K0*n_dict['$TiO_2$']

fig, axs=plt.subplots()
fig.tight_layout()
legend=[]
for err1 in [-0.02, 0.02]:
    for err2 in [-0.02, 0.02]:
        Gbplus=Gcminus*np.exp(-2j*k*d_TiO2*(1-err1))
        Zb=Z_inf_TiO2*(1+Gbplus)/(1-Gbplus)
        Gbminus=(Zb-Z_inf_SiO2)/(Zb+Z_inf_SiO2)
        k=K0*n_dict['$SiO_2$']
        Gaplus=Gbminus*np.exp(-2j*k*d_SiO2*(1-err2))
        Za=Z_inf_SiO2*(1+Gaplus)/(1-Gaplus)
        Gaminus=(Za-Z0)/(Za+Z0)
        R_DLARC_tolerance=np.abs(Gaminus)**2*100
        axs.plot(lambdas,R_DLARC_tolerance)
        legend.append('$TiO_2$ ('+str(err1)+'\%), $SiO_2$ ('+str(err2)+'\%)')
axs.plot(lambdas,R_DLARC_SiO2TiO2)
legend.append('$TiO_2$, $SiO_2$ nominal tickness')
axs.minorticks_on()
axs.set_xlabel('$\lambda$ [nm]')
axs.set_ylabel('Reflectivity [%]')
axs.grid(True,'both')
axs.legend(legend,loc='lower right')
tikzplotlib_fix_ncols(fig)
tikzplotlib.save('Assignment1/DLARC_tol.tex',axis_width='0.9\\textwidth',axis_height ='7cm')

# %% DLARC - tolerance effect
Z_inf_SiO2=Z0/n_dict['$SiO_2$']
Z_inf_TiO2=Z0/n_dict['$TiO_2$']
Gcminus=(Z_inf_Si-Z_inf_TiO2)/(Z_inf_Si+Z_inf_TiO2)
k=K0*n_dict['$TiO_2$']

fig, axs=plt.subplots()
fig.tight_layout()
legend=[]
for err1 in np.linspace(-0.02,0.02,50):
    for err2 in np.linspace(-0.02,0.02,50):
        Gbplus=Gcminus*np.exp(-2j*k*d_TiO2*(1-err1))
        Zb=Z_inf_TiO2*(1+Gbplus)/(1-Gbplus)
        Gbminus=(Zb-Z_inf_SiO2)/(Zb+Z_inf_SiO2)
        k=K0*n_dict['$SiO_2$']
        Gaplus=Gbminus*np.exp(-2j*k*d_SiO2*(1-err2))
        Za=Z_inf_SiO2*(1+Gaplus)/(1-Gaplus)
        Gaminus=(Za-Z0)/(Za+Z0)
        R_DLARC_tolerance=np.abs(Gaminus)**2*100
        axs.plot(lambdas,R_DLARC_tolerance,'gray')
        #legend.append('$TiO_2$ ('+str(err1)+'%), $SiO_2$ ('+str(err2)+'%)')
axs.plot(lambdas,R_DLARC_SiO2TiO2)
legend.append('$TiO_2$, $SiO_2$ nominal tickness')
axs.minorticks_on()
axs.set_xlabel('$\lambda$ [nm]')
axs.set_ylabel('Reflectivity [%]')
axs.grid(True,'both')
axs.legend(legend,loc='lower right')
tikzplotlib_fix_ncols(fig)
tikzplotlib.save('Assignment1/DLARC_tol.tex',axis_width='0.9\\textwidth',axis_height ='7cm')

# %%
plt.show()