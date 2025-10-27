======================================================================
✅ FUSION COMPLETE
======================================================================

📊 Summary:
   Total clips fused: 50,000
   YAMNet dimensions: 1024
   Ecoacoustic dimensions: 17
   Fusion vector size: 1041 dimensions

💾 Output saved to: data/features/embeds_fused_50k/PAPCA

📂 Output structure by logger/date:
   ✓ 2802: 69 dates, 8,367 clips
   ✓ 2823: 202 dates, 41,633 clips

🔍 Sample fused feature structure:

   Metadata columns: ['filepath', 'start_s', 'end_s', 'logger', 'date']
   YAMNet columns: yamnet_0 ... yamnet_1023
   Ecoacoustic columns (17): ['spectral_centroid_mean', 'spectral_centroid_std', 'spectral_bandwidth_mean', 'spectral_rolloff_mean', 'spectral_flatness_mean']... (showing first 5)

   Total columns: 1046
   Shape: (89, 1046)

   Sample row (metadata + first 3 yamnet + first 3 eco):
                                                                         0
filepath                 /home/sparch/Uni-stuff/semester-2/applied_Ml/r...
logger                                                                2802
date                                                              20080226
start_s                                                               20.0
yamnet_0                                                               0.0
yamnet_1                                                          0.000598
yamnet_2                                                               0.0
spectral_centroid_mean                                            0.121941
spectral_centroid_std                                              0.00591
spectral_bandwidth_mean                                            0.10323