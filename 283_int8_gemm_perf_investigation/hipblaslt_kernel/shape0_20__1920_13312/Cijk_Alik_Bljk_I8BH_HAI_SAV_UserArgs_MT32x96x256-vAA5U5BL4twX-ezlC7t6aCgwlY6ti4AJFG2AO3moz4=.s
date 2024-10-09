
/******************************************/
/* Begin Kernel                           */
/******************************************/
.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
.text
.protected Cijk_Alik_Bljk_I8BH_HAI_SAV_UserArgs_MT32x96x256_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_EPS0_GRVWA16_GRVWB16_GSUAMB_IU1_K1_LBSPPA256_LBSPPB256_LBSPPM0_LPA32_LPB32_LPM0_LRVW16_LWPMn1_MIAV1_MIWT1_3_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NEPBS16_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VWA1_VWB1_WSGRA0_WSGRB0_WG32_8_1
.globl Cijk_Alik_Bljk_I8BH_HAI_SAV_UserArgs_MT32x96x256_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_EPS0_GRVWA16_GRVWB16_GSUAMB_IU1_K1_LBSPPA256_LBSPPB256_LBSPPM0_LPA32_LPB32_LPM0_LRVW16_LWPMn1_MIAV1_MIWT1_3_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NEPBS16_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VWA1_VWB1_WSGRA0_WSGRB0_WG32_8_1
.p2align 8
.type Cijk_Alik_Bljk_I8BH_HAI_SAV_UserArgs_MT32x96x256_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_EPS0_GRVWA16_GRVWB16_GSUAMB_IU1_K1_LBSPPA256_LBSPPB256_LBSPPM0_LPA32_LPB32_LPM0_LRVW16_LWPMn1_MIAV1_MIWT1_3_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NEPBS16_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VWA1_VWB1_WSGRA0_WSGRB0_WG32_8_1,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel Cijk_Alik_Bljk_I8BH_HAI_SAV_UserArgs_MT32x96x256_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_EPS0_GRVWA16_GRVWB16_GSUAMB_IU1_K1_LBSPPA256_LBSPPB256_LBSPPM0_LPA32_LPB32_LPM0_LRVW16_LWPMn1_MIAV1_MIWT1_3_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NEPBS16_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VWA1_VWB1_WSGRA0_WSGRB0_WG32_8_1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset 256 // accvgpr offset
  .amdhsa_next_free_vgpr 256 // vgprs
  .amdhsa_next_free_sgpr 76 // sgprs
  .amdhsa_group_segment_fixed_size 36864 // lds bytes
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
  .amdhsa_user_sgpr_count 13
  .amdhsa_user_sgpr_kernarg_preload_length 11
  .amdhsa_user_sgpr_kernarg_preload_offset 0
.end_amdhsa_kernel
.text
/* Num VGPR   =256 */
/* Num AccVGPR=0 */
/* Num SGPR   =76 */

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 4 x 3 */
/* SubGroup= 8 x 32 */
/* VectorWidthA=1 */
/* VectorWidthB=1 */
/* GlobalReadVectorWidthA=16, GlobalReadVectorWidthB=16 */
/* DirectToLdsA=False */
/* DirectToLdsB=False */
/* UseSgprForGRO=1 */
.amdgpu_metadata
---
custom.config:
  InternalSupportParams:
    KernArgsVersion: 2
amdhsa.version:
  - 1
  - 1
amdhsa.kernels:
  - .name: Cijk_Alik_Bljk_I8BH_HAI_SAV_UserArgs_MT32x96x256_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_EPS0_GRVWA16_GRVWB16_GSUAMB_IU1_K1_LBSPPA256_LBSPPB256_LBSPPM0_LPA32_LPB32_LPM0_LRVW16_LWPMn1_MIAV1_MIWT1_3_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NEPBS16_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VWA1_VWB1_WSGRA0_WSGRB0_WG32_8_1
    .symbol: 'Cijk_Alik_Bljk_I8BH_HAI_SAV_UserArgs_MT32x96x256_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_EPS0_GRVWA16_GRVWB16_GSUAMB_IU1_K1_LBSPPA256_LBSPPB256_LBSPPM0_LPA32_LPB32_LPM0_LRVW16_LWPMn1_MIAV1_MIWT1_3_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NEPBS16_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VWA1_VWB1_WSGRA0_WSGRB0_WG32_8_1.kd'
    .language:                   OpenCL C
    .language_version:
      - 2
      - 0
    .args:
      - .name:            Gemm info
        .size:            4
        .offset:          0
        .value_kind:      by_value
        .value_type:      u32
      - .name:            kernel info0
        .size:            4
        .offset:          4
        .value_kind:      by_value
        .value_type:      u32
      - .name:            kernel info1
        .size:            4
        .offset:          8
        .value_kind:      by_value
        .value_type:      u32
      - .name:            numWG
        .size:            4
        .offset:          12
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree0
        .size:            4
        .offset:          16
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree1
        .size:            4
        .offset:          20
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesFree2
        .size:            4
        .offset:          24
        .value_kind:      by_value
        .value_type:      u32
      - .name:            SizesSum0
        .size:            4
        .offset:          28
        .value_kind:      by_value
        .value_type:      u32
      - .name:            D
        .size:            8
        .offset:          32
        .value_kind:      global_buffer
        .value_type:      i8
        .address_space:   generic
      - .name:            C
        .size:            8
        .offset:          40
        .value_kind:      global_buffer
        .value_type:      i8
        .address_space:   generic
      - .name:            A
        .size:            8
        .offset:          48
        .value_kind:      global_buffer
        .value_type:      i8
        .address_space:   generic
      - .name:            B
        .size:            8
        .offset:          56
        .value_kind:      global_buffer
        .value_type:      i8
        .address_space:   generic
      - .name:            strideD0
        .size:            4
        .offset:          64
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideD1
        .size:            4
        .offset:          68
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC0
        .size:            4
        .offset:          72
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideC1
        .size:            4
        .offset:          76
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA0
        .size:            4
        .offset:          80
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideA1
        .size:            4
        .offset:          84
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB0
        .size:            4
        .offset:          88
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB1
        .size:            4
        .offset:          92
        .value_kind:      by_value
        .value_type:      u32
      - .name:            alpha
        .size:            4
        .offset:          96
        .value_kind:      by_value
        .value_type:      i32
      - .name:            beta
        .size:            4
        .offset:          100
        .value_kind:      by_value
        .value_type:      i32
      - .name:            AddressScaleAlphaVec
        .size:            8
        .offset:          104
        .value_kind:      global_buffer
        .value_type:      i32
        .address_space:   generic
      - .name:            activationAlpha
        .size:            4
        .offset:          112
        .value_kind:      by_value
        .value_type:      i32
      - .name:            activationBeta
        .size:            4
        .offset:          116
        .value_kind:      by_value
        .value_type:      i32
      - .name:            activationType
        .size:            4
        .offset:          120
        .value_kind:      by_value
        .value_type:      u32
    .group_segment_fixed_size:   36864
    .kernarg_segment_align:      8
    .kernarg_segment_size:       128
    .max_flat_workgroup_size:    256
    .private_segment_fixed_size: 0
    .sgpr_count:                 76
    .sgpr_spill_count:           0
    .vgpr_count:                 256
    .vgpr_spill_count:           0
    .wavefront_size:             64
...
.end_amdgpu_metadata
Cijk_Alik_Bljk_I8BH_HAI_SAV_UserArgs_MT32x96x256_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_EPS0_GRVWA16_GRVWB16_GSUAMB_IU1_K1_LBSPPA256_LBSPPB256_LBSPPM0_LPA32_LPB32_LPM0_LRVW16_LWPMn1_MIAV1_MIWT1_3_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NEPBS16_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VWA1_VWB1_WSGRA0_WSGRB0_WG32_8_1:
label_ASM_Start:  /// Main body of the asm kernel

/* Magic div and mod functions */
.macro V_MAGIC_DIV dstIdx:req dividend:req magicNumber:req magicShift:req magicA:req
    v_mul_hi_u32 v[\dstIdx+1] \dividend \magicNumber
    v_mul_lo_u32 v[\dstIdx+0] \dividend \magicA
    v_add_u32 v[\dstIdx+0] v[\dstIdx+0] v[\dstIdx+1]
    v_lshrrev_b32 v[\dstIdx+0] \magicShift v[\dstIdx+0]
.endm

/******************************************/
/* VGPR Assignments                       */
/******************************************/
/* ValuC range: [0-12), serializedStore enabled */
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprValuA_X0_I0, 12
.set vgprValuA_X1_I0, 14
.set vgprValuA_X2_I0, 16
.set vgprValuA_X3_I0, 18
.set vgprValuA_X4_I0, 20
.set vgprValuA_X5_I0, 22
.set vgprValuA_X6_I0, 24
.set vgprValuA_X7_I0, 26
.set vgprValuB_X0_I0, 28
.set vgprValuB_X1_I0, 34
.set vgprValuB_X2_I0, 40
.set vgprValuB_X3_I0, 46
.set vgprValuB_X4_I0, 52
.set vgprValuB_X5_I0, 58
.set vgprValuB_X6_I0, 64
.set vgprValuB_X7_I0, 70
.set vgprLocalWriteAddrA, 76
.set vgprLocalWriteAddrB, 77
.set vgprGlobalReadOffsetA, 78
.set vgprGlobalReadOffsetB, 79
.set vgprG2LA, 80
.set vgprG2LB, 88
.set vgprLocalReadAddrA, 112
.set vgprLocalReadAddrB, 113
.set vgprSerial, 114

/******************************************/
/* SGPR Assignments                       */
/******************************************/
.set sgprKernArgAddress, 0
.set sgprWorkGroup0, 2
.set sgprWorkGroup1, 3
.set sgprWorkGroup2, 4
.set sgprArgType, 5
.set sgprGSUSumIdx, 6
.set sgprGSULog2BpeC, 8
.set sgprGSULog2BpeD, 9
.set sgprStaggerU, 10
.set sgprWGM, 11
.set sgprLoopCounterL, 12
.set sgprOrigLoopCounter, 13
.set sgprSrdD, 16
.set sgprSrdC, 20
.set sgprNumWorkGroups0, 14
.set sgprNumWorkGroups1, 15
.set sgprSizesFree, 24
.set sgprSizesSum, 27
.set sgprAddressD, 28
.set sgprAddressC, 30
.set sgprAddressA, 32
.set sgprAddressB, 34
.set sgprStridesD, 36
.set sgprStridesC, 38
.set sgprStridesA, 40
.set sgprStridesB, 42
.set sgprAlpha, 44
.set sgprBeta, 45
.set sgprGSU, 46

/* Size Assignments */
.set sgprSizeI, sgprSizesFree+0
.set sgprSizeJ, sgprSizesFree+1
.set sgprSizeK, sgprSizesFree+2
.set sgprSizeL, sgprSizesSum+0

/* Stride Assignments */
.set constStrideD0I, 1
.set sgprStrideD1J, sgprStridesD+0
.set sgprStrideDK, sgprStridesD+1
.set constStrideC0I, 1
.set sgprStrideC1J, sgprStridesC+0
.set sgprStrideCK, sgprStridesC+1
.set constStrideAL, 1
.set sgprStrideA0I, sgprStridesA+0
.set sgprStrideAK, sgprStridesA+1
.set constStrideBL, 1
.set sgprStrideB1J, sgprStridesB+0
.set sgprStrideBK, sgprStridesB+1

.set MT0, 32
.set MT1, 96
.set DepthU, 256
.set BpeA, 1
.set BpeALog2, 0
.set BpeB, 1
.set BpeBLog2, 0
.set BpeAGR, 1
.set BpeAGRLog2, 0
.set BpeBGR, 1
.set BpeBGRLog2, 0
/* Number of elements to shift-left SRD */
.set SrdShiftLeftA, 16
.set SrdShiftLeftB, 16
/* 2GB limit - set offsets to -1 to exceed this and clamp */
.set BufferLimit, 0xffffffff
.set BufferOOB, 0x80000000

/******************************************/
/* Bits 127:96 of SRD.                    */
/* hex: 0x00020000                        */
/* dst_sel_x (3b): 0                      */
/* dst_sel_y (3b): 0                      */
/* dst_sel_z (3b): 0                      */
/* dst_sel_w (3b): 0                      */
/* num_format (3b): 0                     */
/* data_format (4b): 4                    */
/* user_vm_enable (1b): 0                 */
/* user_vm_mode (1b): 0                   */
/* index_stride (2b): 0                   */
/* add_tid_enable (1b): 0                 */
/* _unusedA (3b): 0                       */
/* nv (1b): 0                             */
/* _unusedB (2b): 0                       */
/* type (2b): 0                           */
/******************************************/
.set Srd127_96, 0x00020000

/* Global Offset A */
.macro GLOBAL_OFFSET_A vgprAddr:req vgprOffsetL:req vgprOffset0I:req vgprTmp:req
    v_mul_lo_u32 v[\vgprTmp+0] s[sgprStrideA0I] v[\vgprOffset0I] // mul d1 lower
    v_add_co_u32 v[\vgprAddr+0] vcc v[\vgprOffsetL] v[\vgprTmp+0] // accumulate K lower
    v_add_u32 v[\vgprAddr+0] 0x10 v[\vgprAddr+0]     // add prepad for pointer shift
                                                       // offset *= bytes/element (multiplier is 1 do nothing)
.endm

/* Global Offset B */
.macro GLOBAL_OFFSET_B vgprAddr:req vgprOffsetL:req vgprOffset1J:req vgprTmp:req
    v_mul_lo_u32 v[\vgprTmp+0] s[sgprStrideB1J] v[\vgprOffset1J] // mul d1 lower
    v_add_co_u32 v[\vgprAddr+0] vcc v[\vgprOffsetL] v[\vgprTmp+0] // accumulate K lower
    v_add_u32 v[\vgprAddr+0] 0x10 v[\vgprAddr+0]     // add prepad for pointer shift
                                                       // offset *= bytes/element (multiplier is 1 do nothing)
.endm

/* Dynamic Scalar Divide: vQuotient=vDividend/vDivisor; vRemainder=vDividend%vDivisor; */
.macro DYNAMIC_VECTOR_DIVIDE vQuotient vRemainder vDividend vDivisor vTmp0 vTmp1 sTmp
    v_cvt_f32_u32 v[\vQuotient] v[\vDivisor]
    v_rcp_f32 v[\vQuotient] v[\vQuotient]
    v_mul_f32 v[\vQuotient] 0x4f800000 v[\vQuotient]
    v_cvt_u32_f32 v[\vQuotient] v[\vQuotient]
    v_mul_lo_u32 v[\vRemainder] v[\vDivisor] v[\vQuotient]
    v_mul_hi_u32 v[\vTmp0] v[\vDivisor] v[\vQuotient]
    v_sub_co_u32 v[\vTmp1] vcc 0x0 v[\vRemainder]
    v_cmp_ne_i32 s[\sTmp:\sTmp+1] 0x0 v[\vTmp0]
    v_cndmask_b32 v[\vRemainder] v[\vTmp1] v[\vRemainder] s[\sTmp:\sTmp+1]
    v_mul_hi_u32 v[\vRemainder] v[\vRemainder] v[\vQuotient]
    v_sub_co_u32 v[\vTmp0] vcc v[\vQuotient] v[\vRemainder]
    v_add_co_u32 v[\vQuotient] vcc v[\vQuotient] v[\vRemainder]
    v_cndmask_b32 v[\vQuotient] v[\vQuotient] v[\vTmp0] s[\sTmp:\sTmp+1]
    v_mul_hi_u32 v[\vQuotient] v[\vQuotient] v[\vDividend]
    v_mul_lo_u32 v[\vRemainder] v[\vQuotient] v[\vDivisor]
    v_sub_co_u32 v[\vTmp0] vcc v[\vDividend] v[\vRemainder]
    v_cmp_ge_u32 s[\sTmp:\sTmp+1] v[\vDividend] v[\vRemainder]
    v_add_co_u32 v[\vRemainder] vcc 0x1 v[\vQuotient]
    v_add_co_u32 v[\vTmp1] vcc -1 v[\vQuotient]
    v_cmp_le_u32 vcc v[\vDivisor] v[\vTmp0]
    s_and_b64 vcc s[\sTmp:\sTmp+1] vcc
    v_cndmask_b32 v[\vQuotient] v[\vQuotient] v[\vRemainder] vcc
    v_cndmask_b32 v[\vQuotient] v[\vTmp1] v[\vQuotient] s[\sTmp:\sTmp+1]
    v_cmp_ne_i32 vcc 0x0 v[\vDivisor]
    v_cndmask_b32 v[\vQuotient] -1 v[\vQuotient] vcc // final result
    v_mul_lo_u32 v[\vRemainder] v[\vQuotient] v[\vDivisor]
    v_sub_co_u32 v[\vRemainder] vcc v[\vDividend] v[\vRemainder] // final result
.endm

/******************************************/
/* Allocate Resources                     */
/******************************************/

/* Load num of Gemms */
s_load_dword s47, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x0

/* Load packed kernel args (StaggerU/GSU) */
s_load_dword s49, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x4

/* Load WGM data */
s_load_dword s[sgprWGM], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x8

/* Load num of WGs */
s_load_dword s50, s[sgprKernArgAddress:sgprKernArgAddress+1], 0xc
s_waitcnt lgkmcnt(0)
s_lshr_b32 s48, s47, 0x1e                          // Get arg type
s_and_b32 s47, 0x3fffffff, s47                     // Get nums of gemm
s_cmp_eq_u32 s48, 0                                // Is kernel args
s_cbranch_scc0 label_HBMArgs
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 0x10 // Shift common args
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0x0

/* Load Kernel Args */
s_load_dwordx16 s[24:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x0
s_load_dwordx4 s[40:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x40
s_load_dwordx2 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x50
s_waitcnt lgkmcnt(0)
s_branch label_LoadArgsEnd
label_HBMArgs:

/* Load address of kernel arguments */
s_load_dwordx2 s[sgprKernArgAddress:sgprKernArgAddress+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x10
s_waitcnt lgkmcnt(0)                               // wait for args to load
label_LoadArgsEnd:
s_branch label_common_kernel_entry

/* pad 37 snops to satisfy 0x100 code size for Preload Backward Compatibility Prologue */
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
label_Preload_Offset_Start:
s_and_b32 s47, 0x3fffffff, s2                      // Get nums of gemm
s_lshr_b32 s48, s2, 0x1e                           // Get arg type
s_mov_b32 s49, s3                                  // Preload internal args
s_cmp_eq_u32 s48, 0                                // Is kernel args
s_cbranch_scc0 label_Preload_HBMArgs
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], 0x10 // Shift common args
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0x0

/* Load Kernel Args */
s_load_dword s31, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x1c
s_load_dwordx8 s[32:39], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x20
s_load_dwordx4 s[40:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x40
s_load_dwordx2 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x50
s_mov_b32 s24, s6                                  // move preload data to correct sgpr
s_mov_b32 s25, s7                                  // move preload data to correct sgpr
s_mov_b32 s26, s8                                  // move preload data to correct sgpr
s_mov_b32 s27, s9                                  // move preload data to correct sgpr
s_mov_b32 s28, s10                                 // move preload data to correct sgpr
s_mov_b32 s29, s11                                 // move preload data to correct sgpr
s_mov_b32 s30, s12                                 // move preload data to correct sgpr
s_branch label_Preload_LoadArgsEnd
label_Preload_HBMArgs:
s_mov_b64 s[sgprKernArgAddress:sgprKernArgAddress+1], s[6:7] // Load address of kernel arguments
label_Preload_LoadArgsEnd:
s_mov_b32 s[sgprWGM], s4                           // Preload internal args2
s_mov_b32 s50, s5                                  // Load num of WGs
label_common_kernel_entry:  /// for both preload/non-preload common code
s_mov_b32 s[sgprWorkGroup0+0], s13                 // restore workgroup id
s_mov_b32 s[sgprWorkGroup0+1], s14                 // restore workgroup id
s_mov_b32 s[sgprWorkGroup0+2], s15                 // restore workgroup id
s_and_b32 s[sgprStaggerU], s49, 0xffff0000         // Restore StaggerU related vars
s_lshr_b32 s[sgprStaggerU], s[sgprStaggerU], 0x10
s_and_b32 s[sgprGSU], s49, 0xffff                  // Restore GSUConfig and GSU
s_mov_b32 s[sgprArgType], s48
s_mov_b32 m0, 0x9000                               // LDS clamp at 36864 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id

/* remap workgroup to XCCs */
s_lshr_b32 s56, s[sgprWGM], 0x10                   // Get WGMXCC
s_ff1_i32_b32 s56, s56                             // Get log(WGMXCC)
s_lshr_b32 s57, s[sgprWGM], 0x16                   // Get CU_Count
/* remap WGs if WGMXCC > 1 ( log(WGMXCC) > 0 ) */
s_cmp_gt_i32 s56, 0
s_cbranch_scc0 label_skip_WGMXCC
/* only remap WGs in the range */
s_lshr_b32 s53, s50, s56
s_lshl_b32 s53, s53, s56
s_cmp_ge_u32 s[sgprWorkGroup0], s53
s_cbranch_scc1 label_skip_WGMXCC
s_cmp_eq_u32 s57, 0                                // CU_Count == 0 ?
s_cbranch_scc0 label_XCCG_nonzero
s_lshr_b32 s53, s[sgprWorkGroup0], s56
s_bfm_b32 s54, s56, 0
s_and_b32 s54, s[sgprWorkGroup0], s54
s_lshr_b32 s55, s50, s56
s_mul_i32 s54, s54, s55
s_add_u32 s[sgprWorkGroup0], s53, s54
s_branch label_skip_WGMXCC
label_XCCG_nonzero:
/* temp0 = (wg//CU_Count)*CU_Count */
v_cvt_f32_u32 v6, s57                              // wg//CU_Count
v_rcp_iflag_f32 v6, v6                             // wg//CU_Count
v_cvt_f32_u32 v7, s[sgprWorkGroup0]                // wg//CU_Count
v_mul_f32 v6, v6, v7                               // wg//CU_Count
v_cvt_u32_f32 v6, v6                               // wg//CU_Count
v_mul_u32_u24 v7, v6, s57                          // wg//CU_Count
v_sub_u32 v7, s[sgprWorkGroup0], v7                // wg//CU_Count
v_cmpx_eq_u32 exec, v7, s57                        // wg//CU_Count
v_add_u32 v6, 1, v6                                // wg//CU_Count
v_mov_b32 v7, 0                                    // wg//CU_Count
s_mov_b64 exec, -1                                 // wg//CU_Count
v_readfirstlane_b32 s53, v6                        // quotient
v_readfirstlane_b32 s54, v7                        // remainder
s_mul_i32 s53, s53, s57
/* temp1 = (wg%CU_Count)//WGMXCC */
s_lshr_b32 s54, s54, s56
/* temp0 = temp0 + temp1 */
s_add_u32 s53, s53, s54
/* temp1 = (wg%WGMXCC) * ((WGs - (WGs//CU_Count) * CU_Count) if (wg > (WGs//CU_Count) * CU_Count) else CU_Count)//WGMXCC */
v_cvt_f32_u32 v6, s57                              // WGs//CU_Count
v_rcp_iflag_f32 v6, v6                             // WGs//CU_Count
v_cvt_f32_u32 v7, s50                              // WGs//CU_Count
v_mul_f32 v6, v6, v7                               // WGs//CU_Count
v_cvt_u32_f32 v6, v6                               // WGs//CU_Count
v_mul_u32_u24 v7, v6, s57                          // WGs//CU_Count
v_sub_u32 v7, s50, v7                              // WGs//CU_Count
v_cmpx_eq_u32 exec, v7, s57                        // WGs//CU_Count
v_add_u32 v6, 1, v6                                // WGs//CU_Count
s_mov_b64 exec, -1                                 // WGs//CU_Count
v_readfirstlane_b32 s54, v6                        // quotient
s_mul_i32 s54, s54, s57
s_sub_u32 s55, s50, s54
s_cmp_gt_u32 s[sgprWorkGroup0], s54
s_cselect_b32 s54, s55, s57
s_lshr_b32 s54, s54, s56
s_bfm_b32 s55, s56, 0
s_and_b32 s55, s[sgprWorkGroup0], s55
s_mul_i32 s54, s54, s55
/* WorkGroup0 = temp0 + temp1 */
s_add_u32 s[sgprWorkGroup0], s53, s54
label_skip_WGMXCC:  /// skip WGMXCC if no enough WGs to remap
s_cmp_eq_u32 s48, 0
s_cbranch_scc0 label_MultiGemm
/* init: add vgpr [12...88) to pool */
/* init: add vgpr [0...12) to pool */
/* init: add agpr [0...0) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
/* lr0I */
v_and_b32 v1, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v0, 15, v1                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v0, 0x8, v0                          // 1. N offset: nOffset = nIdx * nStride(256)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v1, 4, v1                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshlrev_b32 v1, 0x4, v1                          // 5. K offset: lrKOffset = kIdx * mStride(16)
v_add_u32 v0, v1, v0                               // 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v4, 6, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v4, 1, v4                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshlrev_b32 v4, 0xc, v4                          // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(4096)
v_add_u32 v0, v4, v0                               // 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
v_and_b32 v2, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 0x8, v1                          // 1. N offset: nOffset = nIdx * nStride(256)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v2, 4, v2                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshlrev_b32 v2, 0x4, v2                          // 5. K offset: lrKOffset = kIdx * mStride(16)
v_add_u32 v1, v2, v1                               // 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v3, 7, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(128)
v_and_b32 v3, 1, v3                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshlrev_b32 v3, 0xc, v3                          // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(4096)
v_add_u32 v1, v3, v1                               // 7. final local read offset: flrOffset = lrOffset + WOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v2, 6, v[vgprSerial]                 // v2 = v[vgprSerial] / 64
v_lshrrev_b32 v2, 2, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s49, 256                                 // LSU offset: stride = lsuStride(256) when umlds==True
v_mul_lo_u32 v2, s49, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_u32 v[vgprLocalReadAddrA], v2, v0            // Final Offset: offset = (lro0+lsuoffset)*bpeDS(1)
v_lshrrev_b32 v3, 8, v[vgprLocalReadAddrA]         // Final Offset: padding 32 per block 256
v_lshlrev_b32 v3, 0x5, v3                          // Final Offset: padding 32 per block 256
v_add_u32 v[vgprLocalReadAddrA], v3, v[vgprLocalReadAddrA] // Final Offset: add padding 32 per block 256

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 6, v[vgprSerial]                 // v0 = v[vgprSerial] / 64
v_lshrrev_b32 v0, 2, v0                            // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(256) when umlds==True (dup assign opt.)
v_mul_lo_u32 v0, s49, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_u32 v[vgprLocalReadAddrB], v0, v1            // Final Offset: offset = (lro1+lsuoffset)*bpeDS(1)
v_lshrrev_b32 v2, 8, v[vgprLocalReadAddrB]         // Final Offset: padding 32 per block 256
v_lshlrev_b32 v2, 0x5, v2                          // Final Offset: padding 32 per block 256
v_add_u32 v[vgprLocalReadAddrB], v2, v[vgprLocalReadAddrB] // Final Offset: add padding 32 per block 256

/* local read addresses: declare addresses a */
/* N/A */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x2400, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 16 */
/* v1 = A-unroll = serial%LVCA */
v_lshrrev_b32 v0, 4, v[vgprSerial]                 // v0 = v[vgprSerial] / 16
v_and_b32 v1, 15, v[vgprSerial]                    // v1 = v[vgprSerial] % 16
/* unroll *= glvw */
v_lshlrev_b32 v1, 0x4, v1                          // v1 = v1 * 16
v_mov_b32 v4, v1                                   // copy for GlobalSplitU
/* LVCB = 16 */
/* v3 = B-unroll = serial%LVCB */
v_lshrrev_b32 v2, 4, v[vgprSerial]                 // v2 = v[vgprSerial] / 16
v_and_b32 v3, 15, v[vgprSerial]                    // v3 = v[vgprSerial] % 16
/* unroll *= glvw */
v_lshlrev_b32 v3, 0x4, v3                          // v3 = v3 * 16
v_mov_b32 v5, v3                                   // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v4 */
/* lwaUnrollAssignmentB = v5 */

/* local write addresses: first offset a */
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x100, v0    // lwAL**(DepthU_Compute + PAD)
v_add_u32 v[vgprLocalWriteAddrA], v4, v[vgprLocalWriteAddrA] // lwFOA = (lwAA + lwAL*(DepthU+PAD))*bpeDS(1)
v_lshrrev_b32 v6, 8, v[vgprLocalWriteAddrA]        // padding 32 per block 256
v_lshlrev_b32 v6, 0x5, v6                          // padding 32 per block 256
v_add_u32 v[vgprLocalWriteAddrA], v6, v[vgprLocalWriteAddrA] // add padding 32 per block 256

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x100, v2    // lwBL**(DepthU_Compute + PAD)
v_add_u32 v[vgprLocalWriteAddrB], v5, v[vgprLocalWriteAddrB] // lwFOB = (lwBB + lwBL*(DepthU+PAD))*bpeDS(1)
v_lshrrev_b32 v6, 8, v[vgprLocalWriteAddrB]        // padding 32 per block 256
v_lshlrev_b32 v6, 0x5, v6                          // padding 32 per block 256
v_add_u32 v[vgprLocalWriteAddrB], v6, v[vgprLocalWriteAddrB] // add padding 32 per block 256
v_add_co_u32 v[vgprLocalWriteAddrB], vcc, 0x2400, v[vgprLocalWriteAddrB] // lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=9216
v_mov_b32 v8, MT0                                  // set MT0 into sgpr
v_mov_b32 v7, s[sgprSizesFree+0]                   // set Free0 size
v_cvt_f32_u32 v6, v8                               // v6 = ceil(v7 / v8)
v_rcp_iflag_f32 v6, v6                             // v6 = ceil(v7 / v8)
v_cvt_f32_u32 v9, v7                               // v6 = ceil(v7 / v8)
v_mul_f32 v6, v6, v9                               // v6 = ceil(v7 / v8)
v_cvt_u32_f32 v6, v6                               // v6 = ceil(v7 / v8)
v_mul_u32_u24 v9, v6, v8                           // v6 = ceil(v7 / v8)
v_sub_u32 v9, v7, v9                               // v6 = ceil(v7 / v8)
v_cmp_ne_u32 vcc, v9, 0                            // v6 = ceil(v7 / v8)
v_addc_co_u32 v6, vcc, v6, 0, vcc                  // ceil
v_mov_b32 v8, MT1                                  // set MT1 into sgpr
v_mov_b32 v7, s[sgprSizesFree+1]                   // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v6      // set back to numWorkGroup0
v_cvt_f32_u32 v6, v8                               // v6 = ceil(v7 / v8)
v_rcp_iflag_f32 v6, v6                             // v6 = ceil(v7 / v8)
v_cvt_f32_u32 v9, v7                               // v6 = ceil(v7 / v8)
v_mul_f32 v6, v6, v9                               // v6 = ceil(v7 / v8)
v_cvt_u32_f32 v6, v6                               // v6 = ceil(v7 / v8)
v_mul_u32_u24 v9, v6, v8                           // v6 = ceil(v7 / v8)
v_sub_u32 v9, v7, v9                               // v6 = ceil(v7 / v8)
v_cmp_ne_u32 vcc, v9, 0                            // v6 = ceil(v7 / v8)
v_addc_co_u32 v6, vcc, v6, 0, vcc                  // ceil
s_nop 0                                            // 1 wait states
v_readfirstlane_b32 s[sgprNumWorkGroups1], v6      // set back to numWorkGroup1
s_waitcnt lgkmcnt(0)                               // wait for 44/0 bytes of kern args

/* remap wg from 1D(idxWG012) to 3D(wg2,wg1,wg0) */
/* wg2 = idxWG012 * smallMagicNumber(1/(numWG0*numWG1)) */
s_mul_i32 s48, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_and_b32 s49, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s48, s48, s49
v_cvt_f32_u32 v6, s48                              // s48 = s[sgprWorkGroup0] / s48
v_rcp_iflag_f32 v6, v6                             // s48 = s[sgprWorkGroup0] / s48
v_cvt_f32_u32 v7, s[sgprWorkGroup0]                // s48 = s[sgprWorkGroup0] / s48
v_mul_f32 v6, v6, v7                               // s48 = s[sgprWorkGroup0] / s48
v_cvt_u32_f32 v6, v6                               // s48 = s[sgprWorkGroup0] / s48
v_mul_u32_u24 v7, v6, s48                          // s48 = s[sgprWorkGroup0] / s48
v_sub_u32 v7, s[sgprWorkGroup0], v7                // s48 = s[sgprWorkGroup0] / s48
v_cmpx_eq_u32 exec, v7, s48                        // s48 = s[sgprWorkGroup0] / s48
v_add_u32 v6, 1, v6                                // s48 = s[sgprWorkGroup0] / s48
s_mov_b64 exec, -1                                 // s48 = s[sgprWorkGroup0] / s48
v_readfirstlane_b32 s48, v6                        // quotient
s_mov_b32 s[sgprWorkGroup2], s48
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s48, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s48, s48, s[sgprWorkGroup2]
s_mul_i32 s48, s48, s49
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s48
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v6, s[sgprNumWorkGroups0]            // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v6, v6                             // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_f32_u32 v7, s[sgprWorkGroup0]                // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_f32 v6, v6, v7                               // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_u32_f32 v6, v6                               // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_u32_u24 v7, v6, s[sgprNumWorkGroups0]        // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_u32 v7, s[sgprWorkGroup0], v7                // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmpx_eq_u32 exec, v7, s[sgprNumWorkGroups0]      // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_add_u32 v6, 1, v6                                // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b64 exec, -1                                 // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_readfirstlane_b32 s48, v6                        // quotient
s_mov_b32 s[sgprWorkGroup1], s48
/* wg0 = idxWG01 - wg1 * numWG0 */
s_mul_i32 s48, s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s48
s_branch label_MultiGemmEnd
label_MultiGemm:

/* Check if custom structure pointer is null */
s_cmp_eq_u32 s[sgprArgType], 2                     // ArgType == 2 ?
s_cbranch_scc1 label_IsExternalValid               // branch if ArgType == 2
s_mov_b32 s15, 108
s_mul_i32 s54, s47, 4
s_mov_b64 s[48:49], s[sgprKernArgAddress:sgprKernArgAddress+1]
s_branch label_IsExternalValidEnd
label_IsExternalValid:
s_mov_b32 s15, 196
s_mov_b32 s54, 0x0
s_mov_b64 s[48:49], s[sgprKernArgAddress:sgprKernArgAddress+1]
label_IsExternalValidEnd:

/* Grouped Gemm:: prefetch 1 arg load */
s_mov_b32 s14, 1
s_mov_b32 s55, 0
s_load_dwordx4 s[24:27], s[48:49], s54
s_cmpk_eq_u32 s47, 1                               // if gemm_count is 1?
s_cbranch_scc1 label_wgTable_noLoadLoop

/* Grouped Gemm:: accumulate numTiles for each gemm */
/* Grouped Gemm:: loop start */
label_Loop_GemmCount:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s52, s24, 5                             // s52 = s24 / 32
s_and_b32 s50, 31, s24                             // s50 = s24 % 32
s_addc_u32 s52, s52, 0x0
s_mov_b32 s51, 0x0                                 // STATIC_DIV: divisior=96
s_mul_i32 s50, 0x555, s25                          // tmp1 = dividend * magic hi
s_lshl_b64 s[50:51], s[50:51], 0x10                // left shift 16 bits
s_mul_i32 s53, s25, 0x5556                         // tmp0 = dividend * magic lo
s_add_u32 s50, s53, s50                            // add lo
s_addc_u32 s51, s51, 0x0                           // add hi
s_lshr_b64 s[50:51], s[50:51], 0x21                // tmp0 = quotient
s_mul_i32 s51, s50, 0x60                           // tmp1 = quotient * divisor
s_cmp_lg_u32 s51, s25                              // if (quotient * divisor != dividend), result+=1
s_addc_u32 s53, s50, 0x0                           // if (quotient * divisor != dividend), result+=1
s_mul_i32 s52, s52, s53
s_mul_i32 s52, s52, s26
s_and_b32 s53, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s52, s52, s53
s_add_u32 s55, s55, s52
s_cmp_lt_u32 s[sgprWorkGroup0], s55
s_cbranch_scc1 label_FOUND
s_add_u32 s54, s54, s15
s_load_dwordx4 s[24:27], s[48:49], s54
s_add_u32 s14, s14, 1
s_cmp_lt_u32 s14, s47
s_cbranch_scc1 label_Loop_GemmCount

/* Grouped Gemm:: noLoadLoop */
label_wgTable_noLoadLoop:
s_waitcnt lgkmcnt(0)
s_lshr_b32 s52, s24, 5                             // s52 = s24 / 32
s_and_b32 s50, 31, s24                             // s50 = s24 % 32
s_addc_u32 s52, s52, 0x0
s_mov_b32 s51, 0x0                                 // STATIC_DIV: divisior=96
s_mul_i32 s50, 0x555, s25                          // tmp1 = dividend * magic hi
s_lshl_b64 s[50:51], s[50:51], 0x10                // left shift 16 bits
s_mul_i32 s53, s25, 0x5556                         // tmp0 = dividend * magic lo
s_add_u32 s50, s53, s50                            // add lo
s_addc_u32 s51, s51, 0x0                           // add hi
s_lshr_b64 s[50:51], s[50:51], 0x21                // tmp0 = quotient
s_mul_i32 s51, s50, 0x60                           // tmp1 = quotient * divisor
s_cmp_lg_u32 s51, s25                              // if (quotient * divisor != dividend), result+=1
s_addc_u32 s53, s50, 0x0                           // if (quotient * divisor != dividend), result+=1
s_mul_i32 s52, s52, s53
s_mul_i32 s52, s52, s26
s_and_b32 s48, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s52, s52, s48
s_add_u32 s55, s55, s52

/* Grouped Gemm:: gemmIndex found */
label_FOUND:
s_sub_u32 s49, s14, 1
s_sub_u32 s48, s55, s52
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s48
/* Check if custom structure pointer is null */
s_cmp_eq_u32 s[sgprArgType], 2                     // ArgType == 2 ?
s_cbranch_scc1 label_LoadExternalStruct            // branch if ArgType == 2

/* Grouped Gemm: offset argument address to gemm */
/* Grouped Gemm: offset address from wg_table_start to args_start */
s_lshl2_add_u32 s[sgprKernArgAddress], s47, s[sgprKernArgAddress]
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0x0
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s49, s49, 108
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s49
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0x0

/* Load Kernel Args */
s_load_dwordx16 s[28:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x10
s_load_dwordx2 s[44:45], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x50
s_branch label_LoadExternalStructEnd
label_LoadExternalStruct:
/* Grouped Gemm: offset address from args_start to gemm_start */
s_mul_i32 s49, s49, 196
s_add_u32 s[sgprKernArgAddress], s[sgprKernArgAddress], s49
s_addc_u32 s[sgprKernArgAddress+1], s[sgprKernArgAddress+1], 0x0
s_load_dwordx16 s[28:43], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x10
s_load_dword s44, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x50
// Read Beta
s_load_dword s45, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x60
label_LoadExternalStructEnd:
/* init: add vgpr [12...88) to pool */
/* init: add vgpr [0...12) to pool */
/* init: add agpr [0...0) to pool */

/******************************************/
/* Local Read Addresses                   */
/******************************************/

/* local read addresses: tile assignments a/b */
/* lr0I */
v_and_b32 v1, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v0, 15, v1                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v0, 0x8, v0                          // 1. N offset: nOffset = nIdx * nStride(256)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v1, 4, v1                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshlrev_b32 v1, 0x4, v1                          // 5. K offset: lrKOffset = kIdx * mStride(16)
v_add_u32 v0, v1, v0                               // 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v4, 6, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v4, 1, v4                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshlrev_b32 v4, 0xc, v4                          // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(4096)
v_add_u32 v0, v4, v0                               // 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
v_and_b32 v2, 63, v[vgprSerial]                    // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v1, 15, v2                               // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v1, 0x8, v1                          // 1. N offset: nOffset = nIdx * nStride(256)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v2, 4, v2                            // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshlrev_b32 v2, 0x4, v2                          // 5. K offset: lrKOffset = kIdx * mStride(16)
v_add_u32 v1, v2, v1                               // 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v3, 7, v[vgprSerial]                 // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(128)
v_and_b32 v3, 1, v3                                // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshlrev_b32 v3, 0xc, v3                          // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(4096)
v_add_u32 v1, v3, v1                               // 7. final local read offset: flrOffset = lrOffset + WOffset

/* local read addresses: final offsets a */
v_lshrrev_b32 v2, 6, v[vgprSerial]                 // v2 = v[vgprSerial] / 64
v_lshrrev_b32 v2, 2, v2                            // LSU offset: Get LSU wave_id
s_mov_b32 s49, 256                                 // LSU offset: stride = lsuStride(256) when umlds==True
v_mul_lo_u32 v2, s49, v2                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_u32 v[vgprLocalReadAddrA], v2, v0            // Final Offset: offset = (lro0+lsuoffset)*bpeDS(1)
v_lshrrev_b32 v3, 8, v[vgprLocalReadAddrA]         // Final Offset: padding 32 per block 256
v_lshlrev_b32 v3, 0x5, v3                          // Final Offset: padding 32 per block 256
v_add_u32 v[vgprLocalReadAddrA], v3, v[vgprLocalReadAddrA] // Final Offset: add padding 32 per block 256

/* local read addresses: final offsets b */
v_lshrrev_b32 v0, 6, v[vgprSerial]                 // v0 = v[vgprSerial] / 64
v_lshrrev_b32 v0, 2, v0                            // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(256) when umlds==True (dup assign opt.)
v_mul_lo_u32 v0, s49, v0                           // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_u32 v[vgprLocalReadAddrB], v0, v1            // Final Offset: offset = (lro1+lsuoffset)*bpeDS(1)
v_lshrrev_b32 v2, 8, v[vgprLocalReadAddrB]         // Final Offset: padding 32 per block 256
v_lshlrev_b32 v2, 0x5, v2                          // Final Offset: padding 32 per block 256
v_add_u32 v[vgprLocalReadAddrB], v2, v[vgprLocalReadAddrB] // Final Offset: add padding 32 per block 256

/* local read addresses: declare addresses a */
/* N/A */

/* local read addresses: declare addresses b */
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x2400, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)

/******************************************/
/* Local Write Addresses                  */
/******************************************/
/* LVCA = 16 */
/* v1 = A-unroll = serial%LVCA */
v_lshrrev_b32 v0, 4, v[vgprSerial]                 // v0 = v[vgprSerial] / 16
v_and_b32 v1, 15, v[vgprSerial]                    // v1 = v[vgprSerial] % 16
/* unroll *= glvw */
v_lshlrev_b32 v1, 0x4, v1                          // v1 = v1 * 16
v_mov_b32 v4, v1                                   // copy for GlobalSplitU
/* LVCB = 16 */
/* v3 = B-unroll = serial%LVCB */
v_lshrrev_b32 v2, 4, v[vgprSerial]                 // v2 = v[vgprSerial] / 16
v_and_b32 v3, 15, v[vgprSerial]                    // v3 = v[vgprSerial] % 16
/* unroll *= glvw */
v_lshlrev_b32 v3, 0x4, v3                          // v3 = v3 * 16
v_mov_b32 v5, v3                                   // copy for GlobalSplitU
/* lwaUnrollAssignmentA = v4 */
/* lwaUnrollAssignmentB = v5 */

/* local write addresses: first offset a */
v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x100, v0    // lwAL**(DepthU_Compute + PAD)
v_add_u32 v[vgprLocalWriteAddrA], v4, v[vgprLocalWriteAddrA] // lwFOA = (lwAA + lwAL*(DepthU+PAD))*bpeDS(1)
v_lshrrev_b32 v6, 8, v[vgprLocalWriteAddrA]        // padding 32 per block 256
v_lshlrev_b32 v6, 0x5, v6                          // padding 32 per block 256
v_add_u32 v[vgprLocalWriteAddrA], v6, v[vgprLocalWriteAddrA] // add padding 32 per block 256

/* local write addresses: first offset b */
v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x100, v2    // lwBL**(DepthU_Compute + PAD)
v_add_u32 v[vgprLocalWriteAddrB], v5, v[vgprLocalWriteAddrB] // lwFOB = (lwBB + lwBL*(DepthU+PAD))*bpeDS(1)
v_lshrrev_b32 v6, 8, v[vgprLocalWriteAddrB]        // padding 32 per block 256
v_lshlrev_b32 v6, 0x5, v6                          // padding 32 per block 256
v_add_u32 v[vgprLocalWriteAddrB], v6, v[vgprLocalWriteAddrB] // add padding 32 per block 256
v_add_co_u32 v[vgprLocalWriteAddrB], vcc, 0x2400, v[vgprLocalWriteAddrB] // lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=9216
v_mov_b32 v8, MT0                                  // set MT0 into sgpr
v_mov_b32 v7, s[sgprSizesFree+0]                   // set Free0 size
v_cvt_f32_u32 v6, v8                               // v6 = ceil(v7 / v8)
v_rcp_iflag_f32 v6, v6                             // v6 = ceil(v7 / v8)
v_cvt_f32_u32 v9, v7                               // v6 = ceil(v7 / v8)
v_mul_f32 v6, v6, v9                               // v6 = ceil(v7 / v8)
v_cvt_u32_f32 v6, v6                               // v6 = ceil(v7 / v8)
v_mul_u32_u24 v9, v6, v8                           // v6 = ceil(v7 / v8)
v_sub_u32 v9, v7, v9                               // v6 = ceil(v7 / v8)
v_cmp_ne_u32 vcc, v9, 0                            // v6 = ceil(v7 / v8)
v_addc_co_u32 v6, vcc, v6, 0, vcc                  // ceil
v_mov_b32 v8, MT1                                  // set MT1 into sgpr
v_mov_b32 v7, s[sgprSizesFree+1]                   // set Free1 size
v_readfirstlane_b32 s[sgprNumWorkGroups0], v6      // set back to numWorkGroup0
v_cvt_f32_u32 v6, v8                               // v6 = ceil(v7 / v8)
v_rcp_iflag_f32 v6, v6                             // v6 = ceil(v7 / v8)
v_cvt_f32_u32 v9, v7                               // v6 = ceil(v7 / v8)
v_mul_f32 v6, v6, v9                               // v6 = ceil(v7 / v8)
v_cvt_u32_f32 v6, v6                               // v6 = ceil(v7 / v8)
v_mul_u32_u24 v9, v6, v8                           // v6 = ceil(v7 / v8)
v_sub_u32 v9, v7, v9                               // v6 = ceil(v7 / v8)
v_cmp_ne_u32 vcc, v9, 0                            // v6 = ceil(v7 / v8)
v_addc_co_u32 v6, vcc, v6, 0, vcc                  // ceil
s_nop 0                                            // 1 wait states
v_readfirstlane_b32 s[sgprNumWorkGroups1], v6      // set back to numWorkGroup1
s_waitcnt lgkmcnt(0)                               // wait for 44/0 bytes of kern args

/* Early stop if N(SizeFreeJ) == 0 */
s_cmp_eq_u32 s[sgprSizeJ], 0x0
s_cbranch_scc0 label_NoEarlyStop_N0
label_EarlyStop_if_N_is_0:
s_endpgm
label_NoEarlyStop_N0:

/* remap wg from 1D(idxWG012) to 3D(wg2,wg1,wg0) */
/* wg2 = idxWG012 * smallMagicNumber(1/(numWG0*numWG1)) */
s_mul_i32 s48, s[sgprNumWorkGroups0], s[sgprNumWorkGroups1]
s_and_b32 s49, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s48, s48, s49
v_cvt_f32_u32 v6, s48                              // s48 = s[sgprWorkGroup0] / s48
v_rcp_iflag_f32 v6, v6                             // s48 = s[sgprWorkGroup0] / s48
v_cvt_f32_u32 v7, s[sgprWorkGroup0]                // s48 = s[sgprWorkGroup0] / s48
v_mul_f32 v6, v6, v7                               // s48 = s[sgprWorkGroup0] / s48
v_cvt_u32_f32 v6, v6                               // s48 = s[sgprWorkGroup0] / s48
v_mul_u32_u24 v7, v6, s48                          // s48 = s[sgprWorkGroup0] / s48
v_sub_u32 v7, s[sgprWorkGroup0], v7                // s48 = s[sgprWorkGroup0] / s48
v_cmpx_eq_u32 exec, v7, s48                        // s48 = s[sgprWorkGroup0] / s48
v_add_u32 v6, 1, v6                                // s48 = s[sgprWorkGroup0] / s48
s_mov_b64 exec, -1                                 // s48 = s[sgprWorkGroup0] / s48
v_readfirstlane_b32 s48, v6                        // quotient
s_mov_b32 s[sgprWorkGroup2], s48
/* idxWG01 = idxWG012 - wg2 * numWG0 * numWG1 */
s_mul_i32 s48, s[sgprNumWorkGroups1], s[sgprNumWorkGroups0]
s_mul_i32 s48, s48, s[sgprWorkGroup2]
s_mul_i32 s48, s48, s49
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s48
/* wg1 = idxWG01 * smallMagicNumber(1/numWG0) */
v_cvt_f32_u32 v6, s[sgprNumWorkGroups0]            // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_rcp_iflag_f32 v6, v6                             // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_f32_u32 v7, s[sgprWorkGroup0]                // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_f32 v6, v6, v7                               // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cvt_u32_f32 v6, v6                               // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_mul_u32_u24 v7, v6, s[sgprNumWorkGroups0]        // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_sub_u32 v7, s[sgprWorkGroup0], v7                // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_cmpx_eq_u32 exec, v7, s[sgprNumWorkGroups0]      // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_add_u32 v6, 1, v6                                // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
s_mov_b64 exec, -1                                 // s48 = s[sgprWorkGroup0] / s[sgprNumWorkGroups0]
v_readfirstlane_b32 s48, v6                        // quotient
s_mov_b32 s[sgprWorkGroup1], s48
/* wg0 = idxWG01 - wg1 * numWG0 */
s_mul_i32 s48, s[sgprWorkGroup1], s[sgprNumWorkGroups0]
s_sub_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s48

/* Early stop if wg exceed */
s_cmp_ge_u32 s[sgprWorkGroup2], s[sgprSizesFree+2]
s_cbranch_scc0 label_NoEarlyStop_wgExceed
label_EarlyStop_if_wg_exceed:
s_endpgm
label_NoEarlyStop_wgExceed:

label_MultiGemmEnd:
.set sgprSrdA, 48
.set sgprSrdB, 52
.set sgprShadowLimitA, 56
.set sgprShadowLimitB, 58
.set sgprStaggerUIter, 47
.set sgprWrapUA, 60
.set sgprWrapUB, 62
.set sgprGlobalReadIncsA, 64
.set sgprGlobalReadIncsB, 65
.set sgprScalarGlobalReadOffsetA, 66
.set sgprScalarGlobalReadOffsetB, 67
s_sub_u32 s[sgprAddressA+0], s[sgprAddressA+0], 16 // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressA+1], s[sgprAddressA+1], 0 // pre-pad to make room for possible pointer shift
s_sub_u32 s[sgprAddressB+0], s[sgprAddressB+0], 16 // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprAddressB+1], s[sgprAddressB+1], 0 // pre-pad to make room for possible pointer shift

/* Short circuit condition if Alpha == 0, then sumDims=0 */
s_cmp_eq_u32 s[sgprAlpha], 0                       // s[Alpha] == 0 ?
s_cbranch_scc0 label_AlphaNonZero                  // branch if s[Alpha] != 0
s_mov_b32 s[sgprSizesSum+0], 0x0                   // Set summation dim=0 if Alpha == 0
label_AlphaNonZero:

/******************************************/
/* Begin setupNewTile                     */
/******************************************/

/* global read addresses: work-group */
/* graWorkGroup mapping */
s_and_b32 s72, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s72, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU                           // branch if GSU == 1
// GSU-not-WGMapRR :nwg1 = (size1J + MT1J - 1) / MT1J;
s_and_b32 s72, s[sgprGSU], 0x4000                  // SCC = (GSUWGMRR == 1) ?
s_cbranch_scc1 label_GSUWGMRR                      // branch if GSUWGMRR == 1
s_and_b32 s72, s[sgprGSU], 0x3fff                  // Restore GSU
v_cvt_f32_u32 v6, s72                              // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s72
v_rcp_iflag_f32 v6, v6                             // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s72
v_cvt_f32_u32 v7, s[sgprWorkGroup1]                // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s72
v_mul_f32 v6, v6, v7                               // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s72
v_cvt_u32_f32 v6, v6                               // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s72
v_mul_u32_u24 v7, v6, s72                          // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s72
v_sub_u32 v7, s[sgprWorkGroup1], v7                // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s72
v_cmpx_eq_u32 exec, v7, s72                        // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s72
v_add_u32 v6, 1, v6                                // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s72
v_mov_b32 v7, 0                                    // s[sgprGSUSumIdx] = s[sgprWorkGroup1] % s72
s_mov_b64 exec, -1                                 // s[sgprWorkGroup1] = s[sgprWorkGroup1] / s72
v_readfirstlane_b32 s[sgprWorkGroup1], v6          // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx], v7           // remainder
s_branch label_GSUWGMRR_End
label_GSUWGMRR:
v_cvt_f32_u32 v6, s[sgprNumWorkGroups1]            // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_rcp_iflag_f32 v6, v6                             // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_cvt_f32_u32 v7, s[sgprWorkGroup1]                // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_mul_f32 v6, v6, v7                               // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_cvt_u32_f32 v6, v6                               // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_mul_u32_u24 v7, v6, s[sgprNumWorkGroups1]        // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_sub_u32 v7, s[sgprWorkGroup1], v7                // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_cmpx_eq_u32 exec, v7, s[sgprNumWorkGroups1]      // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_add_u32 v6, 1, v6                                // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_mov_b32 v7, 0                                    // s[sgprWorkGroup1] = s[sgprWorkGroup1] % s[sgprNumWorkGroups1]
s_mov_b64 exec, -1                                 // s[sgprGSUSumIdx] = s[sgprWorkGroup1] / s[sgprNumWorkGroups1]
v_readfirstlane_b32 s[sgprGSUSumIdx], v6           // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v7          // remainder
label_GSUWGMRR_End:
s_mov_b32 s[sgprGSULog2BpeC], 0
s_mov_b32 s[sgprGSULog2BpeD], 2
s_branch label_GSU_End
label_GSU:
s_mov_b64 s[sgprGSUSumIdx:sgprGSUSumIdx+1], 0      // Set GSUSumIdx to 0
s_mov_b32 s[sgprGSULog2BpeC], 0
s_mov_b32 s[sgprGSULog2BpeD], 0
label_GSU_End:
s_sext_i32_i16 s[sgprWGM], s[sgprWGM]              // Restore WGM
s_cmp_gt_i32 s[sgprWGM], 1                         // WGM > 1 ?
s_cbranch_scc1 label_WGMPositive                   // branch if WGM > 1
s_cmp_ge_i32 s[sgprWGM], 0                         // WGM >= 0 ?
s_cbranch_scc1 label_WGM                           // branch if WGM >= 0
s_abs_i32 s[sgprWGM], s[sgprWGM]                   // abs(WGM)
v_cvt_f32_u32 v6, s[sgprWGM]                       // WGM
v_rcp_iflag_f32 v6, v6                             // WGM
v_cvt_f32_u32 v7, s[sgprWorkGroup0]                // WGM
v_mul_f32 v6, v6, v7                               // WGM
v_cvt_u32_f32 v6, v6                               // WGM
v_mul_u32_u24 v7, v6, s[sgprWGM]                   // WGM
v_sub_u32 v7, s[sgprWorkGroup0], v7                // WGM
v_cmpx_eq_u32 exec, v7, s[sgprWGM]                 // WGM
v_add_u32 v6, 1, v6                                // WGM
s_mov_b64 exec, -1                                 // WGM
v_readfirstlane_b32 s74, v6                        // quotient
s_mul_i32 s75, s74, s[sgprWGM]                     // quotient * non-magic divisor
s_sub_u32 s75, s[sgprWorkGroup0], s75              // WorkGroup0=remainder
s_mul_i32 s75, s75, s[sgprNumWorkGroups1]          // (wg1 % WGM)*NumWorkGroups1
s_add_u32 s75, s75, s[sgprWorkGroup1]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups1
v_cvt_f32_u32 v6, s[sgprWGM]                       // WGM
v_rcp_iflag_f32 v6, v6                             // WGM
v_cvt_f32_u32 v7, s[sgprNumWorkGroups0]            // WGM
v_mul_f32 v6, v6, v7                               // WGM
v_cvt_u32_f32 v6, v6                               // WGM
v_mul_u32_u24 v7, v6, s[sgprWGM]                   // WGM
v_sub_u32 v7, s[sgprNumWorkGroups0], v7            // WGM
v_cmpx_eq_u32 exec, v7, s[sgprWGM]                 // WGM
v_add_u32 v6, 1, v6                                // WGM
s_mov_b64 exec, -1                                 // WGM
v_readfirstlane_b32 s72, v6                        // quotient
s_mul_i32 s73, s[sgprWGM], s72                     // quotient * non-magic divisor
s_sub_u32 s73, s[sgprNumWorkGroups0], s73          // NumWorkGroups0=remainder
s_cmp_eq_u32 s73, 0                                // remainder == 0 ?
s_cmov_b32 s73, s[sgprWGM]                         // remainder = WGM if remainder == 0
s_cmp_ge_u32 s74, s72                              // blockId >= numFullBlocks ?
s_cselect_b32 s72, s73, s[sgprWGM]
v_cvt_f32_u32 v6, s72                              // s[sgprWorkGroup1] = s75 / s72
v_rcp_iflag_f32 v6, v6                             // s[sgprWorkGroup1] = s75 / s72
v_cvt_f32_u32 v7, s75                              // s[sgprWorkGroup1] = s75 / s72
v_mul_f32 v6, v6, v7                               // s[sgprWorkGroup1] = s75 / s72
v_cvt_u32_f32 v6, v6                               // s[sgprWorkGroup1] = s75 / s72
v_mul_u32_u24 v7, v6, s72                          // s[sgprWorkGroup1] = s75 / s72
v_sub_u32 v7, s75, v7                              // s[sgprWorkGroup1] = s75 / s72
v_cmpx_eq_u32 exec, v7, s72                        // s[sgprWorkGroup1] = s75 / s72
v_add_u32 v6, 1, v6                                // s[sgprWorkGroup1] = s75 / s72
v_mov_b32 v7, 0                                    // s[sgprWorkGroup0] = s75 % s72
s_mov_b64 exec, -1                                 // s[sgprWorkGroup1] = s75 / s72
v_readfirstlane_b32 s[sgprWorkGroup1], v6          // quotient
v_readfirstlane_b32 s[sgprWorkGroup0], v7          // remainder
s_mul_i32 s[sgprWorkGroup0], s[sgprWorkGroup1], s72 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup0], s75, s[sgprWorkGroup0] // WorkGroup0=remainder
s_mul_i32 s74, s74, s[sgprWGM]                     // blockId * WGM
s_add_u32 s[sgprWorkGroup0], s[sgprWorkGroup0], s74 // wg1 += blockId * WGM
s_branch label_WGM
label_WGMPositive:
v_cvt_f32_u32 v6, s[sgprWGM]                       // WGM
v_rcp_iflag_f32 v6, v6                             // WGM
v_cvt_f32_u32 v7, s[sgprWorkGroup1]                // WGM
v_mul_f32 v6, v6, v7                               // WGM
v_cvt_u32_f32 v6, v6                               // WGM
v_mul_u32_u24 v7, v6, s[sgprWGM]                   // WGM
v_sub_u32 v7, s[sgprWorkGroup1], v7                // WGM
v_cmpx_eq_u32 exec, v7, s[sgprWGM]                 // WGM
v_add_u32 v6, 1, v6                                // WGM
s_mov_b64 exec, -1                                 // WGM
v_readfirstlane_b32 s74, v6                        // quotient
s_mul_i32 s75, s74, s[sgprWGM]                     // quotient * non-magic divisor
s_sub_u32 s75, s[sgprWorkGroup1], s75              // WorkGroup1=remainder
s_mul_i32 s75, s75, s[sgprNumWorkGroups0]          // (wg1 % WGM)*NumWorkGroups0
s_add_u32 s75, s75, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*NumWorkGroups0
v_cvt_f32_u32 v6, s[sgprWGM]                       // WGM
v_rcp_iflag_f32 v6, v6                             // WGM
v_cvt_f32_u32 v7, s[sgprNumWorkGroups1]            // WGM
v_mul_f32 v6, v6, v7                               // WGM
v_cvt_u32_f32 v6, v6                               // WGM
v_mul_u32_u24 v7, v6, s[sgprWGM]                   // WGM
v_sub_u32 v7, s[sgprNumWorkGroups1], v7            // WGM
v_cmpx_eq_u32 exec, v7, s[sgprWGM]                 // WGM
v_add_u32 v6, 1, v6                                // WGM
s_mov_b64 exec, -1                                 // WGM
v_readfirstlane_b32 s72, v6                        // quotient
s_mul_i32 s73, s[sgprWGM], s72                     // quotient * non-magic divisor
s_sub_u32 s73, s[sgprNumWorkGroups1], s73          // NumWorkGroups1=remainder
s_cmp_eq_u32 s73, 0                                // remainder == 0 ?
s_cmov_b32 s73, s[sgprWGM]                         // remainder = WGM if remainder == 0
s_cmp_ge_u32 s74, s72                              // blockId >= numFullBlocks ?
s_cselect_b32 s72, s73, s[sgprWGM]
v_cvt_f32_u32 v6, s72                              // s[sgprWorkGroup0] = s75 / s72
v_rcp_iflag_f32 v6, v6                             // s[sgprWorkGroup0] = s75 / s72
v_cvt_f32_u32 v7, s75                              // s[sgprWorkGroup0] = s75 / s72
v_mul_f32 v6, v6, v7                               // s[sgprWorkGroup0] = s75 / s72
v_cvt_u32_f32 v6, v6                               // s[sgprWorkGroup0] = s75 / s72
v_mul_u32_u24 v7, v6, s72                          // s[sgprWorkGroup0] = s75 / s72
v_sub_u32 v7, s75, v7                              // s[sgprWorkGroup0] = s75 / s72
v_cmpx_eq_u32 exec, v7, s72                        // s[sgprWorkGroup0] = s75 / s72
v_add_u32 v6, 1, v6                                // s[sgprWorkGroup0] = s75 / s72
v_mov_b32 v7, 0                                    // s[sgprWorkGroup1] = s75 % s72
s_mov_b64 exec, -1                                 // s[sgprWorkGroup0] = s75 / s72
v_readfirstlane_b32 s[sgprWorkGroup0], v6          // quotient
v_readfirstlane_b32 s[sgprWorkGroup1], v7          // remainder
s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s72 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup1], s75, s[sgprWorkGroup1] // WorkGroup1=remainder
s_mul_i32 s74, s74, s[sgprWGM]                     // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s74 // wg1 += blockId * WGM
label_WGM:

/* global read addresses: tile offset assignment a */
/* graTileAssignmentA = v0 */

/* global read addresses: tile offset assignment b */
/* graTileAssignmentB = v2 */

/* global read addresses: unroll assignment a */
/* v1 */

/* global read addresses: unroll assignment b */
/* v3 */

/* global read addresses: other free assignments */
/* s[sgprWorkGroup2] */

/* global read addresses: tile offsets a */

/* global read addresses: tile offsets b */

/* global read addresses: unroll offsets a */

/* global read addresses: unroll offsets b */

/* global read addresses: final offsets a */
GLOBAL_OFFSET_A vgprGlobalReadOffsetA+0,  1,  0, 6 // gROA_0_0_0_0
s_mul_i32 s[sgprScalarGlobalReadOffsetA+0], s[sgprStrideA0I], 16 // compute offset diff (scaled tileDim)
                                                   // scalar offset *= bytes/element (multiplier is 1, do nothing)

/* global read addresses: final offsets b */
GLOBAL_OFFSET_B vgprGlobalReadOffsetB+0,  3,  2, 6 // gROB_0_0_0_0
s_mul_i32 s[sgprScalarGlobalReadOffsetB+0], s[sgprStrideB1J], 16 // compute offset diff (scaled tileDim)
                                                   // scalar offset *= bytes/element (multiplier is 1, do nothing)
s_mul_i32 s[sgprScalarGlobalReadOffsetB+1], s[sgprStrideB1J], 32 // compute offset diff (scaled tileDim)
                                                   // scalar offset *= bytes/element (multiplier is 1, do nothing)
s_mul_i32 s[sgprScalarGlobalReadOffsetB+2], s[sgprStrideB1J], 48 // compute offset diff (scaled tileDim)
                                                   // scalar offset *= bytes/element (multiplier is 1, do nothing)
s_mul_i32 s[sgprScalarGlobalReadOffsetB+3], s[sgprStrideB1J], 64 // compute offset diff (scaled tileDim)
                                                   // scalar offset *= bytes/element (multiplier is 1, do nothing)
s_mul_i32 s[sgprScalarGlobalReadOffsetB+4], s[sgprStrideB1J], 80 // compute offset diff (scaled tileDim)
                                                   // scalar offset *= bytes/element (multiplier is 1, do nothing)

/* global read addresses: addresses a */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s75, s[sgprWorkGroup0], 32            // WorkGroup[01] * MT
s_mul_i32 s74, s[sgprWorkGroup0], 32               // WorkGroup[01] * MT
s_mul_hi_u32 s75, s74, s[sgprStrideA0I]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s74, s74, s[sgprStrideA0I]               // tlu=0, scaled tile-offset by stride
s_and_b32 s72, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_A                        // branch if GSUC == 1
s_mul_hi_u32 s73, 256, s[sgprGSUSumIdx]            // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s72, 256, s[sgprGSUSumIdx]               // gsuOffset = DepthU*GSUSumIdx
s_branch label_GSUC_A_End
label_GSUC_A:
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum], 8 // s[LoopCounterL] = s[sgprSizesSum] / 256
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0x3fff   // Restore GSU
v_cvt_f32_u32 v0, s[sgprGSUSumIdx+1]               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v0, v0                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v1, s[sgprLoopCounterL]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v0, v0, v1                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v0, v0                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v1, v0, s[sgprGSUSumIdx+1]           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_u32 v1, s[sgprLoopCounterL], v1              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmpx_eq_u32 exec, v1, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_u32 v0, 1, v0                                // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v1, 0                                    // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b64 exec, -1                                 // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_readfirstlane_b32 s[sgprLoopCounterL], v0        // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v1         // remainder
s_mul_i32 s73, s[sgprLoopCounterL], s[sgprGSUSumIdx] // quotient*GSUSumIdx
s_add_u32 s72, 1, s[sgprLoopCounterL]              // quotient+1
s_add_u32 s73, s73, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s72, s72, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s72, s72, s73                        // (quotient+1)*GSUSumIdx if needed
s_mul_hi_u32 s73, s72, 256                         // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
s_mul_i32 s72, s72, 256                            // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
label_GSUC_A_End:
s_add_u32 s74, s74, s72                            // accum GsuOffset term to tilestart
s_addc_u32 s75, s75, s73                           // accum GsuOffset term to tilestart
s_mov_b32 s[sgprShadowLimitA+0], 1                 // Init tensor size
s_mov_b32 s[sgprShadowLimitA+1], 0                 // init tensor size
s_sub_u32 s72, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s73, constStrideAL, s72               // stride x (size-1)
s_mul_i32 s72, constStrideAL, s72                  // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s72 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s73 // sum tensor size
s_sub_u32 s72, s[sgprSizeI], 1                     // (size-1)
s_mul_hi_u32 s73, s[sgprStrideA0I], s72            // stride x (size-1)
s_mul_i32 s72, s[sgprStrideA0I], s72               // stride x (size-1)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s72 // sum tensor size
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s73 // sum tensor size
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s74 // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s75 // sub tileStart
                                                   // Set limit to use bytes (byte is 1, do nothing)
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s73, s[sgprStrideAK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s72, s[sgprStrideAK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s74, s74, s72                            // accum wg term to tilestart
s_addc_u32 s75, s75, s73                           // accum wg term to tilestart
                                                   // tileStart *= BPE (multiplier is 1, do nothing)
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s74    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s75   // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: addresses b */
/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s75, s[sgprWorkGroup1], 96            // WorkGroup[01] * MT
s_mul_i32 s74, s[sgprWorkGroup1], 96               // WorkGroup[01] * MT
s_mul_hi_u32 s75, s74, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s74, s74, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_and_b32 s72, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_B                        // branch if GSUC == 1
s_mul_hi_u32 s73, 256, s[sgprGSUSumIdx]            // gsuOffset = DepthU*GSUSumIdx
s_mul_i32 s72, 256, s[sgprGSUSumIdx]               // gsuOffset = DepthU*GSUSumIdx
s_branch label_GSUC_B_End
label_GSUC_B:
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum], 8 // s[LoopCounterL] = s[sgprSizesSum] / 256
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0x3fff   // Restore GSU
v_cvt_f32_u32 v0, s[sgprGSUSumIdx+1]               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v0, v0                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v1, s[sgprLoopCounterL]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v0, v0, v1                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v0, v0                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v1, v0, s[sgprGSUSumIdx+1]           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_u32 v1, s[sgprLoopCounterL], v1              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmpx_eq_u32 exec, v1, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_u32 v0, 1, v0                                // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v1, 0                                    // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b64 exec, -1                                 // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_readfirstlane_b32 s[sgprLoopCounterL], v0        // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v1         // remainder
s_mul_i32 s73, s[sgprLoopCounterL], s[sgprGSUSumIdx] // quotient*GSUSumIdx
s_add_u32 s72, 1, s[sgprLoopCounterL]              // quotient+1
s_add_u32 s73, s73, s[sgprGSUSumIdx+1]             // quotient*GSUSumIdx+remainder
s_mul_i32 s72, s72, s[sgprGSUSumIdx]               // (quotient+1)*GSUSumIdx
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cselect_b32 s72, s72, s73                        // (quotient+1)*GSUSumIdx if needed
s_mul_hi_u32 s73, s72, 256                         // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
s_mul_i32 s72, s72, 256                            // gsuOffset = DepthU*accumulatedNumOfLoopCounterL
label_GSUC_B_End:
s_add_u32 s74, s74, s72                            // accum GsuOffset term to tilestart
s_addc_u32 s75, s75, s73                           // accum GsuOffset term to tilestart
s_mov_b32 s[sgprShadowLimitB+0], 1                 // Init tensor size
s_mov_b32 s[sgprShadowLimitB+1], 0                 // init tensor size
s_sub_u32 s72, s[sgprSizeL], 1                     // (size-1)
s_mul_hi_u32 s73, constStrideBL, s72               // stride x (size-1)
s_mul_i32 s72, constStrideBL, s72                  // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s72 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s73 // sum tensor size
s_sub_u32 s72, s[sgprSizeJ], 1                     // (size-1)
s_mul_hi_u32 s73, s[sgprStrideB1J], s72            // stride x (size-1)
s_mul_i32 s72, s[sgprStrideB1J], s72               // stride x (size-1)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s72 // sum tensor size
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s73 // sum tensor size
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s74 // sub tileStart
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s75 // sub tileStart
                                                   // Set limit to use bytes (byte is 1, do nothing)
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 16 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_mul_hi_u32 s73, s[sgprStrideBK], s[sgprWorkGroup2] // Stride*WG
s_mul_i32 s72, s[sgprStrideBK], s[sgprWorkGroup2]  // Stride*WG
s_add_u32 s74, s74, s72                            // accum wg term to tilestart
s_addc_u32 s75, s75, s73                           // accum wg term to tilestart
                                                   // tileStart *= BPE (multiplier is 1, do nothing)
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s74    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s75   // SRD base = Address+ tileStart1
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD

/* global read addresses: increments a */
s_and_b32 s73, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s73, s73, DepthU*BpeAGR                  // GSU*DepthU*Bpe
s_and_b32 s72, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cselect_b32 s[sgprGlobalReadIncsA+0], DepthU*BpeAGR, s73 // incrA (unrollIdx)

/* global read addresses: increments b */
s_and_b32 s73, s[sgprGSU], 0x3fff                  // Restore GSU
s_mul_i32 s73, s73, DepthU*BpeBGR                  // GSU*DepthU*Bpe
s_and_b32 s72, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cselect_b32 s[sgprGlobalReadIncsB+0], DepthU*BpeBGR, s73 // incrB (unrollIdx)
/* declare loop num iterations */
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+0], 8 // s[sgprLoopCounterL] = s[sgprSizesSum+0] / 256
s_and_b32 s72, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s72, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU_1                         // branch if GSU == 1
s_and_b32 s[sgprGSUSumIdx+1], s[sgprGSU], 0x3fff   // Restore GSU
v_cvt_f32_u32 v0, s[sgprGSUSumIdx+1]               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_rcp_iflag_f32 v0, v0                             // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_f32_u32 v1, s[sgprLoopCounterL]              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_f32 v0, v0, v1                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cvt_u32_f32 v0, v0                               // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mul_u32_u24 v1, v0, s[sgprGSUSumIdx+1]           // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_sub_u32 v1, s[sgprLoopCounterL], v1              // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_cmpx_eq_u32 exec, v1, s[sgprGSUSumIdx+1]         // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_add_u32 v0, 1, v0                                // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_mov_b32 v1, 0                                    // s[sgprGSUSumIdx+1] = s[sgprLoopCounterL] % s[sgprGSUSumIdx+1]
s_mov_b64 exec, -1                                 // s[sgprLoopCounterL] = s[sgprLoopCounterL] / s[sgprGSUSumIdx+1]
v_readfirstlane_b32 s[sgprLoopCounterL], v0        // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v1         // remainder
s_add_u32 s72, 1, s[sgprLoopCounterL]              // tmp<-numIterMyWg+1
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], s72                // numIterMyWg++ if needed
label_GSU_1:
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter
s_and_b32 s74, s[sgprStaggerU], 0x1f00
s_lshr_b32 s74, s74, 0x8
s_and_b32 s75, s[sgprStaggerU], 0xe000
s_and_b32 s[sgprStaggerU], s[sgprStaggerU], 0xff
s_mov_b32 s72, s[sgprStaggerU]                     // init staggerU
label_beginStaggerUIter:
s_lshl_b32 s73, s72, s74                           // shift by StaggerUStride
s_cmp_ge_u32 s[sgprOrigLoopCounter], s73           // loopCount >= current shift Count
s_cbranch_scc1 label_endStaggerUIter               // jump to end
s_lshr_b32 s72, s72, 1                             // step down to smaller stagger
s_branch label_beginStaggerUIter                   // jump to begin
label_endStaggerUIter:
s_sub_u32 s73, s72, 1                              // staggerU mask
s_cmp_ge_u32 s72, 1                                // if current staggerU >= 1
s_cselect_b32 s[sgprStaggerUIter], s73, 0          // set Mask
s_cmp_eq_u32 s75, 0x0
s_cbranch_scc1 label_StaggerUMapping_1
s_mov_b32 s72, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping_1:
s_cmp_eq_u32 s75, 0x2000
s_cbranch_scc1 label_StaggerUMapping_2
s_mov_b32 s72, s[sgprWorkGroup1]
s_branch label_staggerInputEnd
label_StaggerUMapping_2:
s_cmp_eq_u32 s75, 0x4000
s_cbranch_scc1 label_StaggerUMapping_3
s_mov_b32 s72, -0x1
s_branch label_staggerInputEnd
label_StaggerUMapping_3:
s_cmp_eq_u32 s75, 0x6000
s_cbranch_scc1 label_StaggerUMapping_4
s_mul_i32 s73, s[sgprNumWorkGroups0], s[sgprWorkGroup1]
s_add_u32 s72, s72, s73
s_add_u32 s72, s72, s[sgprWorkGroup0]
s_branch label_staggerInputEnd
label_StaggerUMapping_4:
s_cmp_eq_u32 s75, 0x8000
s_cbranch_scc1 label_staggerInputEnd
s_mov_b32 s72, -0x1
s_branch label_staggerInputEnd
label_staggerInputEnd:
s_and_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s72 // Compute actual stagger start for this tile
s_lshl_b32 s[sgprStaggerUIter], s[sgprStaggerUIter], s74 // shift by StaggerUStride

/* SRDs += (StaggerUIter) * GlobalReadIncsA+0 */
s_mul_hi_i32 s73, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_i32 s72, s[sgprStaggerUIter], s[sgprGlobalReadIncsA+0] //  stagger byte offset
s_mul_hi_i32 s[sgprWrapUA+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUA+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsA+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0], s[sgprWrapUA+0] // remove one iteration
s_subb_u32 s[sgprWrapUA+1], 0, s[sgprWrapUA+1]     // remove one iteration
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s72        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s73       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s72 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s73 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* SRDs += (StaggerUIter) * GlobalReadIncsB+0 */
s_mul_hi_i32 s73, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_i32 s72, s[sgprStaggerUIter], s[sgprGlobalReadIncsB+0] //  stagger byte offset
s_mul_hi_i32 s[sgprWrapUB+1], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_mul_i32 s[sgprWrapUB+0], s[sgprLoopCounterL], s[sgprGlobalReadIncsB+0] // Number of bytes accessed by the unroll loop
s_sub_u32 s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0], s[sgprWrapUB+0] // remove one iteration
s_subb_u32 s[sgprWrapUB+1], 0, s[sgprWrapUB+1]     // remove one iteration
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s72        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s73       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s72 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s73 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_add_u32 s[sgprStaggerUIter], s[sgprStaggerUIter], 2 // Subtract (PGR-1); StaggerUIter now contains target iteration to wrap
/* local read addresses: init pointers a */

/* localReadInitPointers */
/* local read addresses: init pointers b */

/* localReadInitPointers */

/* prefetch: global -> local */
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?
s_cbranch_scc1 label_ShadowInitStart               // skip to ShadowInitStart iter b/c numIter==0
buffer_load_dwordx4 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LA+4:vgprG2LA+4+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 // G -> Reg 0_0_1_0
buffer_load_dwordx4 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:0 // G -> Reg 0_0_1_0
buffer_load_dwordx4 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+1] offen offset:0 // G -> Reg 0_0_2_0
buffer_load_dwordx4 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+2] offen offset:0 // G -> Reg 0_0_3_0
buffer_load_dwordx4 v[vgprG2LB+16:vgprG2LB+16+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+3] offen offset:0 // G -> Reg 0_0_4_0
buffer_load_dwordx4 v[vgprG2LB+20:vgprG2LB+20+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+4] offen offset:0 // G -> Reg 0_0_5_0

/* global read inc A loopL */
s_add_u32 s74, s[sgprLoopCounterL], 1              // remove pf(1)
s_cmp_eq_u32 s[sgprStaggerUIter], s74              // Is this wrapIter? (pf)
s_cselect_b32 s72, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
s_cselect_b32 s73, s[sgprWrapUA+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s72        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s73       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s72 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s73 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s74, s[sgprLoopCounterL], 1              // remove pf(1)
s_cmp_eq_u32 s[sgprStaggerUIter], s74              // Is this wrapIter? (pf)
s_cselect_b32 s72, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
s_cselect_b32 s73, s[sgprWrapUB+1], 0              // incUpper <- ?
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s72        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s73       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s72 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s73 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

/******************************************/
/* End setupNewTile                       */
/******************************************/
label_ShadowInitStart:
s_mov_b32 s[sgprSrdD+0], s[sgprAddressD+0]         // init SRD base address (lower)
s_mov_b32 s[sgprSrdD+1], s[sgprAddressD+1]         // init SRD base address (upper) + other fields
s_mov_b32 s[sgprSrdD+2], 0x80000000
s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_mov_b32 s[sgprSrdC+0], s[sgprAddressC+0]         // init SRD base address (lower)
s_mov_b32 s[sgprSrdC+1], s[sgprAddressC+1]         // init SRD base address (upper) + other fields
s_mov_b32 s[sgprSrdC+2], 0x80000000
s_mov_b32 s[sgprSrdC+3], Srd127_96                 // Set bits 127_96 in post-loop SRD


s_mul_i32 s74, MT1, s[sgprWorkGroup1]              // <- wg1*MT1
s_mul_hi_u32 s73, s74, s[sgprStrideC1J]            // ScaleC s74 by Stride
s_mul_i32 s72, s74, s[sgprStrideC1J]               // ScaleC s74 by Stride
s_lshl_b64 s[72:73], s[72:73], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprAddressC+0], s72    // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprAddressC+1], s73   // add hi to SRD
s_mul_hi_u32 s73, s74, s[sgprStrideD1J]            // ScaleD s74 by Stride
s_mul_i32 s72, s74, s[sgprStrideD1J]               // ScaleD s74 by Stride
s_lshl_b64 s[72:73], s[72:73], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprAddressD+0], s72    // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprAddressD+1], s73   // add hi to SRD

s_mul_hi_u32 s73, s[sgprWorkGroup2], s[sgprStrideCK] // ScaleC s[sgprWorkGroup2] by Stride
s_mul_i32 s72, s[sgprWorkGroup2], s[sgprStrideCK]  // ScaleC s[sgprWorkGroup2] by Stride
s_lshl_b64 s[72:73], s[72:73], s[sgprGSULog2BpeC]  // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s72        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s73       // add hi to SRD
s_mul_hi_u32 s73, s[sgprWorkGroup2], s[sgprStrideDK] // ScaleD s[sgprWorkGroup2] by Stride
s_mul_i32 s72, s[sgprWorkGroup2], s[sgprStrideDK]  // ScaleD s[sgprWorkGroup2] by Stride
s_lshl_b64 s[72:73], s[72:73], s[sgprGSULog2BpeD]  // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s72        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s73       // add hi to SRD

s_and_b32 s72, s[sgprGSU], 0x3fff                  // Restore GSU
s_cmp_eq_u32 s72, 1                                // GSU == 1 ?
s_cbranch_scc1 label_GSU_2                         // branch if GSU == 1
// GSU Output Buffer offset: Free0 + (Free1-1)*StrideC1J + (Free2-1)*StrideCK * GSUIdx * bpe%s
s_mul_hi_u32 s73, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_mul_i32 s72, s[sgprSizesFree+0], s[sgprGSUSumIdx] // Free0
s_sub_u32 s74, s[sgprSizesFree+1], 1               // Free1
s_mul_i32 s74, s74, s[sgprGSUSumIdx]               // Free1
s_mul_hi_u32 s75, s74, s[sgprStrideC1J]            // Free1
s_mul_i32 s74, s74, s[sgprStrideC1J]               // Free1
s_add_u32 s72, s72, s74                            // Free1
s_addc_u32 s73, s73, s75                           // Free1
s_sub_u32 s74, s[sgprSizesFree+2], 1               // Free2
s_mul_i32 s74, s74, s[sgprGSUSumIdx]               // Free2
s_mul_hi_u32 s75, s74, s[sgprStrideCK]             // Free2
s_mul_i32 s74, s74, s[sgprStrideCK]                // Free2
s_add_u32 s72, s72, s74                            // Free2
s_addc_u32 s73, s73, s75                           // Free2
s_lshl_b64 s[72:73], s[72:73], 2                   // scale by bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s72        // add lo GSU offset to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s73       // add hi GSU offset to SRD
label_GSU_2:
.set sgprGSULog2BpeC, UNDEF
.set sgprAddressC, UNDEF
.set sgprAddressD, UNDEF

/* initC: remove ValuC vgpr buffer [0...12) from pool */

/* initC: remove acc vgpr buffer [0...0) from pool */

/* initC: remove ValuA/B vgpr buffer [12...76) from pool */
v_mov_b32 v[vgprValuC+0], 0x0                      // initC
v_mov_b32 v[vgprValuC+1], 0x0                      // initC
v_mov_b32 v[vgprValuC+2], 0x0                      // initC
v_mov_b32 v[vgprValuC+3], 0x0                      // initC
v_mov_b32 v[vgprValuC+4], 0x0                      // initC
v_mov_b32 v[vgprValuC+5], 0x0                      // initC
v_mov_b32 v[vgprValuC+6], 0x0                      // initC
v_mov_b32 v[vgprValuC+7], 0x0                      // initC
v_mov_b32 v[vgprValuC+8], 0x0                      // initC
v_mov_b32 v[vgprValuC+9], 0x0                      // initC
v_mov_b32 v[vgprValuC+10], 0x0                     // initC
v_mov_b32 v[vgprValuC+11], 0x0                     // initC
s_cmp_eq_u32 s[sgprLoopCounterL], 0                // at last iteration?

/* after InitC, skip to end of prefetch last iter if numIter==0 */
s_cbranch_scc0 label_NoBranch_LN50C2C0V5RNYGOT_0   // Only branch on scc1
s_getpc_b64 s[28:29]                               // addr of next instr
s_add_i32 s30, label_PrefetchGlobalLastIterEnd, 0x4 // target branch offset
s_add_u32 s28, s28, s30                            // add target branch offset
s_addc_u32 s29, s29, 0                             // add high and carry
s_setpc_b64 s[28:29]                               // branch to label_PrefetchGlobalLastIterEnd
label_NoBranch_LN50C2C0V5RNYGOT_0:
s_waitcnt vmcnt(0)                                 // 8wait for global read

/* local write a */
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+4:vgprG2LA+4+3] offset:4608 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 4608

/* local write b */
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+4:vgprG2LB+4+3] offset:4608 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 4608
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+8:vgprG2LB+8+3] offset:9216 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 9216
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+12:vgprG2LB+12+3] offset:13824 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 13824
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+16:vgprG2LB+16+3] offset:18432 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 18432
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+20:vgprG2LB+20+3] offset:23040 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 23040

/* local write swap a */

/* local write swap b */
s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // PGR=2 but only 1 loop
s_cbranch_scc1 label_skipPGR2_0                    // PGR=2 but only 1 loop
buffer_load_dwordx4 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LA+4:vgprG2LA+4+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 // G -> Reg 0_0_1_0
buffer_load_dwordx4 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx4 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:0 // G -> Reg 0_0_1_0
buffer_load_dwordx4 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+1] offen offset:0 // G -> Reg 0_0_2_0
buffer_load_dwordx4 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+2] offen offset:0 // G -> Reg 0_0_3_0
buffer_load_dwordx4 v[vgprG2LB+16:vgprG2LB+16+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+3] offen offset:0 // G -> Reg 0_0_4_0
buffer_load_dwordx4 v[vgprG2LB+20:vgprG2LB+20+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+4] offen offset:0 // G -> Reg 0_0_5_0
label_skipPGR2_0:
s_waitcnt lgkmcnt(0)                               // 0prefetch wait for local write
// Skip force waitcnt0
s_barrier

/* local read prefetch a */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read prefetch b */
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:9216 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:18432 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read inc a */
/* N/A, lro->64 */
/* self.localReadDoCntA 1 self.localReadDoCntB 1 */

/* local read inc b */
/* N/A, lro->64 */
/* self.localReadDoCntA 1 self.localReadDoCntB 1 */

/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/
label_openLoopL:
s_cmp_eq_u32 s[sgprLoopCounterL], 0x1              // LoopCounterL < EndCounter
s_cbranch_scc1 label_toPGR1_0                      // PGR=2 but only 1 loop, toPGR1
s_cmp_le_u32 s[sgprLoopCounterL], 0x2              // LoopCounterL < EndCounter
s_cbranch_scc1 label_LoopEndL                      // do not enter LoopL
label_LoopBeginL:

/******************************************/
/* Unrolled Loop 1/1 - Begin              */
/******************************************/

/* Begin Each Unroll: Check VGPR.checkin for INT8 LW */

/* iter 0 */
/*  grEndMfmaIndex:18, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(2)                               // wait for prior local read local write old=0, new=2 newLW=0 newLR=2 for iteration == 0
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:1  */
ds_read_b128 v[vgprValuA_X2_I0+0:vgprValuA_X2_I0+0+3], v[vgprLocalReadAddrA] offset:64 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
ds_read_b128 v[vgprValuB_X2_I0+0:vgprValuB_X2_I0+0+3], v[vgprLocalReadAddrB] offset:64 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_waitcnt lgkmcnt(2)                               // wait for prior local read local write
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:2  */
ds_read_b128 v[vgprValuB_X2_I0+4:vgprValuB_X2_I0+4+3], v[vgprLocalReadAddrB] offset:9280 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
ds_read_b128 v[vgprValuB_X2_I0+8:vgprValuB_X2_I0+8+3], v[vgprLocalReadAddrB] offset:18496 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
s_cselect_b32 s28, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=3 */

/* iter 1 */
/*  grEndMfmaIndex:18, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:3  */
ds_read_b128 v[vgprValuA_X4_I0+0:vgprValuA_X4_I0+0+3], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=128 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
ds_read_b128 v[vgprValuB_X4_I0+0:vgprValuB_X4_I0+0+3], v[vgprLocalReadAddrB] offset:128 // L -> Reg lro=128 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
s_cselect_b32 s29, s[sgprWrapUA+1], 0              // incUpper <- ?
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X0_I0+0+2+0:vgprValuA_X0_I0+0+2+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:4  */
ds_read_b128 v[vgprValuB_X4_I0+4:vgprValuB_X4_I0+4+3], v[vgprLocalReadAddrB] offset:9344 // L -> Reg lro=128 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s28        // gra SRD += inc(lower)
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X0_I0+4+2+0:vgprValuB_X0_I0+4+2+0+1], v[vgprValuA_X0_I0+0+2+0:vgprValuA_X0_I0+0+2+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:5  */
ds_read_b128 v[vgprValuB_X4_I0+8:vgprValuB_X4_I0+8+3], v[vgprLocalReadAddrB] offset:18560 // L -> Reg lro=128 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s29       // gra SRD += inc(upper)
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X0_I0+8+2+0:vgprValuB_X0_I0+8+2+0+1], v[vgprValuA_X0_I0+0+2+0:vgprValuA_X0_I0+0+2+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=2 skipReadsIterA=2 readsPerIterA=1 */
/* dataAtIterB=-1 numReadsIterB=2 skipReadsIterB=2 readsPerIterB=3 */

/* iter 2 */
/*  grEndMfmaIndex:18, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:6  */
ds_read_b128 v[vgprValuA_X6_I0+0:vgprValuA_X6_I0+0+3], v[vgprLocalReadAddrA] offset:192 // L -> Reg lro=192 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
ds_read_b128 v[vgprValuB_X6_I0+0:vgprValuB_X6_I0+0+3], v[vgprLocalReadAddrB] offset:192 // L -> Reg lro=192 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s28 // limit -= inc)
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:7  */
ds_read_b128 v[vgprValuB_X6_I0+4:vgprValuB_X6_I0+4+3], v[vgprLocalReadAddrB] offset:9408 // L -> Reg lro=192 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s29 // limit -= inc)
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:8  */
ds_read_b128 v[vgprValuB_X6_I0+8:vgprValuB_X6_I0+8+3], v[vgprLocalReadAddrB] offset:18624 // L -> Reg lro=192 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X2_I0+8+0+0:vgprValuB_X2_I0+8+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=3 skipReadsIterA=2 readsPerIterA=1 */
/* dataAtIterB=0 numReadsIterB=3 skipReadsIterB=2 readsPerIterB=3 */

/* iter 3 */
/*  grEndMfmaIndex:18, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:9  */
/* localReadsVacancy: latencyLeft 2 */
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=8
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X2_I0+0+2+0:vgprValuA_X2_I0+0+2+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:10  */
/* localReadsVacancy: latencyLeft 2 */

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X2_I0+4+2+0:vgprValuB_X2_I0+4+2+0+1], v[vgprValuA_X2_I0+0+2+0:vgprValuA_X2_I0+0+2+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:11  */
/* localReadsVacancy: latencyLeft 2 */
s_cselect_b32 s28, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X2_I0+8+2+0:vgprValuB_X2_I0+8+2+0+1], v[vgprValuA_X2_I0+0+2+0:vgprValuA_X2_I0+0+2+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=3 skipReadsIterA=2 readsPerIterA=1 */
/* dataAtIterB=0 numReadsIterB=3 skipReadsIterB=2 readsPerIterB=3 */

/* iter 4 */
/*  grEndMfmaIndex:18, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:12  */
/* localReadsVacancy: latencyLeft 2 */
s_cselect_b32 s29, s[sgprWrapUB+1], 0              // incUpper <- ?
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=4
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+1], v[vgprValuA_X4_I0+0+0+0:vgprValuA_X4_I0+0+0+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:13  */
/* localReadsVacancy: latencyLeft 2 */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s28        // gra SRD += inc(lower)
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X4_I0+4+0+0:vgprValuB_X4_I0+4+0+0+1], v[vgprValuA_X4_I0+0+0+0:vgprValuA_X4_I0+0+0+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:14  */
/* localReadsVacancy: latencyLeft 2 */
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s29       // gra SRD += inc(upper)
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X4_I0+8+0+0:vgprValuB_X4_I0+8+0+0+1], v[vgprValuA_X4_I0+0+0+0:vgprValuA_X4_I0+0+0+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=1 numReadsIterB=3 skipReadsIterB=1 readsPerIterB=3 */

/* iter 5 */
/*  grEndMfmaIndex:18, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:15  */
/* localReadsVacancy: latencyLeft 2 */
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s28 // limit -= inc)
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=4
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X4_I0+0+2+0:vgprValuB_X4_I0+0+2+0+1], v[vgprValuA_X4_I0+0+2+0:vgprValuA_X4_I0+0+2+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:16  */
/* localReadsVacancy: latencyLeft 2 */
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s29 // limit -= inc)
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X4_I0+4+2+0:vgprValuB_X4_I0+4+2+0+1], v[vgprValuA_X4_I0+0+2+0:vgprValuA_X4_I0+0+2+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:17  */
/* schedule remaining localreads for 1LDSB */
/* localReadsVacancy: latencyLeft 2 */
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X4_I0+8+2+0:vgprValuB_X4_I0+8+2+0+1], v[vgprValuA_X4_I0+0+2+0:vgprValuA_X4_I0+0+2+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=1 numReadsIterB=3 skipReadsIterB=1 readsPerIterB=3 */

/* iter 6 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:18, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:18  */
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0
buffer_load_dwordx4 v[vgprG2LA+0:vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // G -> Reg 0_0_0_0
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+4:vgprG2LA+4+3] offset:4608 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 4608
buffer_load_dwordx4 v[vgprG2LA+4:vgprG2LA+4+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 // G -> Reg 0_0_1_0
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
buffer_load_dwordx4 v[vgprG2LB+0:vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // G -> Reg 0_0_0_0
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+4:vgprG2LB+4+3] offset:4608 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 4608
buffer_load_dwordx4 v[vgprG2LB+4:vgprG2LB+4+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:0 // G -> Reg 0_0_1_0
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+8:vgprG2LB+8+3] offset:9216 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 9216
buffer_load_dwordx4 v[vgprG2LB+8:vgprG2LB+8+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+1] offen offset:0 // G -> Reg 0_0_2_0
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+12:vgprG2LB+12+3] offset:13824 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 13824
buffer_load_dwordx4 v[vgprG2LB+12:vgprG2LB+12+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+2] offen offset:0 // G -> Reg 0_0_3_0
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+16:vgprG2LB+16+3] offset:18432 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 18432
buffer_load_dwordx4 v[vgprG2LB+16:vgprG2LB+16+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+3] offen offset:0 // G -> Reg 0_0_4_0
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+20:vgprG2LB+20+3] offset:23040 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 23040
buffer_load_dwordx4 v[vgprG2LB+20:vgprG2LB+20+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+4] offen offset:0 // G -> Reg 0_0_5_0

/* local write swap offsets a */

/* local write swap offsets b */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=8 newLR=0
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+1], v[vgprValuA_X6_I0+0+0+0:vgprValuA_X6_I0+0+0+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:19  */
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X6_I0+4+0+0:vgprValuB_X6_I0+4+0+0+1], v[vgprValuA_X6_I0+0+0+0:vgprValuA_X6_I0+0+0+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:20  */

/* local read swap offsets a */

/* local read swap offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X6_I0+8+0+0:vgprValuB_X6_I0+8+0+0+1], v[vgprValuA_X6_I0+0+0+0:vgprValuA_X6_I0+0+0+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=1 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=0 readsPerIterB=3 */

/* iter 7 */
/*  grEndMfmaIndex:18, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:21  */
s_waitcnt lgkmcnt(0)                               // 3wait for local write
// Skip force waitcnt0
s_barrier
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=8 newLR=0
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X6_I0+0+2+0:vgprValuB_X6_I0+0+2+0+1], v[vgprValuA_X6_I0+0+2+0:vgprValuA_X6_I0+0+2+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:22  */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X6_I0+4+2+0:vgprValuB_X6_I0+4+2+0+1], v[vgprValuA_X6_I0+0+2+0:vgprValuA_X6_I0+0+2+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:23  */
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:9216 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:18432 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X6_I0+8+2+0:vgprValuB_X6_I0+8+2+0+1], v[vgprValuA_X6_I0+0+2+0:vgprValuA_X6_I0+0+2+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=1 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=1 readsPerIterB=3 */

/******************************************/
/* Unrolled Loop - End                    */
/******************************************/

/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x2              // counterL==2
s_cbranch_scc0 label_LoopBeginL                    // restart LoopL
label_LoopEndL:

/* Before NLL: Check VGPR.checkin for INT8 LW */

/******************************************/
/* Ord. NoGlobalLoadLoop - Begin          */
/******************************************/

/* iter 0 */
/*  grEndMfmaIndex:18, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(2)                               // wait for prior local read local write old=0, new=2 newLW=0 newLR=2 for iteration == 0
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:1  */
ds_read_b128 v[vgprValuA_X2_I0+0:vgprValuA_X2_I0+0+3], v[vgprLocalReadAddrA] offset:64 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
ds_read_b128 v[vgprValuB_X2_I0+0:vgprValuB_X2_I0+0+3], v[vgprLocalReadAddrB] offset:64 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0

/* global read inc A loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
s_waitcnt lgkmcnt(2)                               // wait for prior local read local write
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:2  */
ds_read_b128 v[vgprValuB_X2_I0+4:vgprValuB_X2_I0+4+3], v[vgprLocalReadAddrB] offset:9280 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
ds_read_b128 v[vgprValuB_X2_I0+8:vgprValuB_X2_I0+8+3], v[vgprLocalReadAddrB] offset:18496 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
s_cselect_b32 s28, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0] // incLower <- ?
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=3 */

/* iter 1 */
/*  grEndMfmaIndex:18, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:3  */
ds_read_b128 v[vgprValuA_X4_I0+0:vgprValuA_X4_I0+0+3], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=128 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
ds_read_b128 v[vgprValuB_X4_I0+0:vgprValuB_X4_I0+0+3], v[vgprLocalReadAddrB] offset:128 // L -> Reg lro=128 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
s_cselect_b32 s29, s[sgprWrapUA+1], 0              // incUpper <- ?
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X0_I0+0+2+0:vgprValuA_X0_I0+0+2+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:4  */
ds_read_b128 v[vgprValuB_X4_I0+4:vgprValuB_X4_I0+4+3], v[vgprLocalReadAddrB] offset:9344 // L -> Reg lro=128 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s28        // gra SRD += inc(lower)
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X0_I0+4+2+0:vgprValuB_X0_I0+4+2+0+1], v[vgprValuA_X0_I0+0+2+0:vgprValuA_X0_I0+0+2+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:5  */
ds_read_b128 v[vgprValuB_X4_I0+8:vgprValuB_X4_I0+8+3], v[vgprLocalReadAddrB] offset:18560 // L -> Reg lro=128 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s29       // gra SRD += inc(upper)
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X0_I0+8+2+0:vgprValuB_X0_I0+8+2+0+1], v[vgprValuA_X0_I0+0+2+0:vgprValuA_X0_I0+0+2+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=2 skipReadsIterA=2 readsPerIterA=1 */
/* dataAtIterB=-1 numReadsIterB=2 skipReadsIterB=2 readsPerIterB=3 */

/* iter 2 */
/*  grEndMfmaIndex:18, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:6  */
ds_read_b128 v[vgprValuA_X6_I0+0:vgprValuA_X6_I0+0+3], v[vgprLocalReadAddrA] offset:192 // L -> Reg lro=192 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
ds_read_b128 v[vgprValuB_X6_I0+0:vgprValuB_X6_I0+0+3], v[vgprLocalReadAddrB] offset:192 // L -> Reg lro=192 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s28 // limit -= inc)
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:7  */
ds_read_b128 v[vgprValuB_X6_I0+4:vgprValuB_X6_I0+4+3], v[vgprLocalReadAddrB] offset:9408 // L -> Reg lro=192 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s29 // limit -= inc)
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:8  */
ds_read_b128 v[vgprValuB_X6_I0+8:vgprValuB_X6_I0+8+3], v[vgprLocalReadAddrB] offset:18624 // L -> Reg lro=192 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X2_I0+8+0+0:vgprValuB_X2_I0+8+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=3 skipReadsIterA=2 readsPerIterA=1 */
/* dataAtIterB=0 numReadsIterB=3 skipReadsIterB=2 readsPerIterB=3 */

/* iter 3 */
/*  grEndMfmaIndex:18, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:9  */
/* localReadsVacancy: latencyLeft 2 */
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=8
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X2_I0+0+2+0:vgprValuA_X2_I0+0+2+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:10  */
/* localReadsVacancy: latencyLeft 2 */

/* global read inc B loopL */
s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter] // Is this the wrapIter?
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X2_I0+4+2+0:vgprValuB_X2_I0+4+2+0+1], v[vgprValuA_X2_I0+0+2+0:vgprValuA_X2_I0+0+2+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:11  */
/* localReadsVacancy: latencyLeft 2 */
s_cselect_b32 s28, s[sgprWrapUB+0], s[sgprGlobalReadIncsB+0] // incLower <- ?
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X2_I0+8+2+0:vgprValuB_X2_I0+8+2+0+1], v[vgprValuA_X2_I0+0+2+0:vgprValuA_X2_I0+0+2+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=3 skipReadsIterA=2 readsPerIterA=1 */
/* dataAtIterB=0 numReadsIterB=3 skipReadsIterB=2 readsPerIterB=3 */

/* iter 4 */
/*  grEndMfmaIndex:18, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:12  */
/* localReadsVacancy: latencyLeft 2 */
s_cselect_b32 s29, s[sgprWrapUB+1], 0              // incUpper <- ?
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=4
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+1], v[vgprValuA_X4_I0+0+0+0:vgprValuA_X4_I0+0+0+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:13  */
/* localReadsVacancy: latencyLeft 2 */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s28        // gra SRD += inc(lower)
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X4_I0+4+0+0:vgprValuB_X4_I0+4+0+0+1], v[vgprValuA_X4_I0+0+0+0:vgprValuA_X4_I0+0+0+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:14  */
/* localReadsVacancy: latencyLeft 2 */
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s29       // gra SRD += inc(upper)
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X4_I0+8+0+0:vgprValuB_X4_I0+8+0+0+1], v[vgprValuA_X4_I0+0+0+0:vgprValuA_X4_I0+0+0+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=1 numReadsIterB=3 skipReadsIterB=1 readsPerIterB=3 */

/* iter 5 */
/*  grEndMfmaIndex:18, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:15  */
/* localReadsVacancy: latencyLeft 2 */
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s28 // limit -= inc)
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=4
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X4_I0+0+2+0:vgprValuB_X4_I0+0+2+0+1], v[vgprValuA_X4_I0+0+2+0:vgprValuA_X4_I0+0+2+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:16  */
/* localReadsVacancy: latencyLeft 2 */
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s29 // limit -= inc)
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X4_I0+4+2+0:vgprValuB_X4_I0+4+2+0+1], v[vgprValuA_X4_I0+0+2+0:vgprValuA_X4_I0+0+2+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:17  */
/* schedule remaining localreads for 1LDSB */
/* localReadsVacancy: latencyLeft 2 */
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X4_I0+8+2+0:vgprValuB_X4_I0+8+2+0+1], v[vgprValuA_X4_I0+0+2+0:vgprValuA_X4_I0+0+2+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=1 numReadsIterB=3 skipReadsIterB=1 readsPerIterB=3 */

/* iter 6 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  */
/*  grEndMfmaIndex:18, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:18  */
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(7)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(6)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+4:vgprG2LA+4+3] offset:4608 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 4608
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(5)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(4)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+4:vgprG2LB+4+3] offset:4608 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 4608
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(3)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+8:vgprG2LB+8+3] offset:9216 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 9216
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(2)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+12:vgprG2LB+12+3] offset:13824 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 13824
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(1)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+16:vgprG2LB+16+3] offset:18432 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 18432
/* sched write - iter 6 writesPerItem=1 */
s_waitcnt vmcnt(0)                                 // wait for global read before writing to local
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+20:vgprG2LB+20+3] offset:23040 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 23040

/* local write swap offsets a */

/* local write swap offsets b */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=8 newLR=0
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+1], v[vgprValuA_X6_I0+0+0+0:vgprValuA_X6_I0+0+0+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:19  */
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X6_I0+4+0+0:vgprValuB_X6_I0+4+0+0+1], v[vgprValuA_X6_I0+0+0+0:vgprValuA_X6_I0+0+0+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:20  */

/* local read swap offsets a */

/* local read swap offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X6_I0+8+0+0:vgprValuB_X6_I0+8+0+0+1], v[vgprValuA_X6_I0+0+0+0:vgprValuA_X6_I0+0+0+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=1 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=0 readsPerIterB=3 */

/* iter 7 */
/*  grEndMfmaIndex:18, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:21  */
s_waitcnt lgkmcnt(0)                               // 3wait for local write
// Skip force waitcnt0
s_barrier
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=8 newLR=0
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X6_I0+0+2+0:vgprValuB_X6_I0+0+2+0+1], v[vgprValuA_X6_I0+0+2+0:vgprValuA_X6_I0+0+2+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:22  */
ds_read_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X6_I0+4+2+0:vgprValuB_X6_I0+4+2+0+1], v[vgprValuA_X6_I0+0+2+0:vgprValuA_X6_I0+0+2+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:23  */
ds_read_b128 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+3], v[vgprLocalReadAddrB] offset:9216 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b128 v[vgprValuB_X0_I0+8:vgprValuB_X0_I0+8+3], v[vgprLocalReadAddrB] offset:18432 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X6_I0+8+2+0:vgprValuB_X6_I0+8+2+0+1], v[vgprValuA_X6_I0+0+2+0:vgprValuA_X6_I0+0+2+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=1 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=1 readsPerIterB=3 */
label_toPGR1_0:
s_and_b32 s8, s[sgprGSU], 0x3fff                   // Restore GSU
s_cmp_eq_u32 s8, 1                                 // GSU == 1 ?
s_cbranch_scc0 label_GSU_3                         // branch if GSU != 1

/******************************************/
/* Opt. NoLoadLoop - Begin                */
/******************************************/
s_cmpk_eq_u32 s[sgprBeta], 0x0                     // Beta == 0
s_cbranch_scc0 label_OptNLL_End                    // Branch if Beta is not zero

s_cmp_eq_u32 s[sgprAlpha], 1                       // Alpha == 1.0 ?
s_cbranch_scc0 label_OptNLL_End                    // branch if alpha != 1

s_and_b32 s28, 31, s[sgprSizeI]                    // s28 = s[sgprSizeI] % 32
s_add_u32 s29, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s29                // wg0 >= nwg0-1 ?
s_cselect_b32 s28, s28, 0                          // set rMT0
s_cmpk_gt_u32 s28, 0x0                             // rMT0 > 0
s_cbranch_scc1 label_OptNLL_End                    // jump if edges required
s_mov_b32 s31, 0x0                                 // STATIC_DIV: divisior=96
s_mul_i32 s30, 0x555, s[sgprSizeJ]                 // tmp1 = dividend * magic hi
s_lshl_b64 s[30:31], s[30:31], 0x10                // left shift 16 bits
s_mul_i32 s29, s[sgprSizeJ], 0x5556                // tmp0 = dividend * magic lo
s_add_u32 s30, s29, s30                            // add lo
s_addc_u32 s31, s31, 0x0                           // add hi
s_lshr_b64 s[30:31], s[30:31], 0x21                // tmp1 = (dividend * magic) << shift
s_mov_b32 s29, s30                                 // quotient
s_mul_i32 s30, s29, 0x60                           // quotient*divisor
s_sub_u32 s28, s[sgprSizeJ], s30                   // rReg = dividend - quotient*divisor
s_add_u32 s29, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s29                // wg1 >= nwg1-1
s_cselect_b32 s28, s28, 0                          // set rMT1
s_cmpk_gt_u32 s28, 0x0                             // rMT1 > 0
s_cbranch_scc1 label_OptNLL_End                    // jump if edges required

s_and_b32 s29, 255, s[sgprSizesSum+0]              // s29 = s[sgprSizesSum+0] % 256
s_cmp_eq_u32 s29, 0x0                              // numIterL == 0
s_cbranch_scc0 label_OptNLL_End                    // skip if tail loop required

/* iter 0 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(2)                               // wait for prior local read local write old=0, new=2 newLW=0 newLR=2 for iteration == 0
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:1  */
ds_read_b128 v[vgprValuA_X2_I0+0:vgprValuA_X2_I0+0+3], v[vgprLocalReadAddrA] offset:64 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
ds_read_b128 v[vgprValuB_X2_I0+0:vgprValuB_X2_I0+0+3], v[vgprLocalReadAddrB] offset:64 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
s_waitcnt lgkmcnt(2)                               // wait for prior local read local write
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:2  */
ds_read_b128 v[vgprValuB_X2_I0+4:vgprValuB_X2_I0+4+3], v[vgprLocalReadAddrB] offset:9280 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
ds_read_b128 v[vgprValuB_X2_I0+8:vgprValuB_X2_I0+8+3], v[vgprLocalReadAddrB] offset:18496 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=3 */

/* iter 1 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:3  */
ds_read_b128 v[vgprValuA_X4_I0+0:vgprValuA_X4_I0+0+3], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=128 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
ds_read_b128 v[vgprValuB_X4_I0+0:vgprValuB_X4_I0+0+3], v[vgprLocalReadAddrB] offset:128 // L -> Reg lro=128 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X0_I0+0+2+0:vgprValuA_X0_I0+0+2+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:4  */
ds_read_b128 v[vgprValuB_X4_I0+4:vgprValuB_X4_I0+4+3], v[vgprLocalReadAddrB] offset:9344 // L -> Reg lro=128 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X0_I0+4+2+0:vgprValuB_X0_I0+4+2+0+1], v[vgprValuA_X0_I0+0+2+0:vgprValuA_X0_I0+0+2+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:5  */
ds_read_b128 v[vgprValuB_X4_I0+8:vgprValuB_X4_I0+8+3], v[vgprLocalReadAddrB] offset:18560 // L -> Reg lro=128 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X0_I0+8+2+0:vgprValuB_X0_I0+8+2+0+1], v[vgprValuA_X0_I0+0+2+0:vgprValuA_X0_I0+0+2+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=2 skipReadsIterA=2 readsPerIterA=1 */
/* dataAtIterB=-1 numReadsIterB=2 skipReadsIterB=2 readsPerIterB=3 */

/* iter 2 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:6  */
ds_read_b128 v[vgprValuA_X6_I0+0:vgprValuA_X6_I0+0+3], v[vgprLocalReadAddrA] offset:192 // L -> Reg lro=192 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
ds_read_b128 v[vgprValuB_X6_I0+0:vgprValuB_X6_I0+0+3], v[vgprLocalReadAddrB] offset:192 // L -> Reg lro=192 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:7  */
ds_read_b128 v[vgprValuB_X6_I0+4:vgprValuB_X6_I0+4+3], v[vgprLocalReadAddrB] offset:9408 // L -> Reg lro=192 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:8  */
ds_read_b128 v[vgprValuB_X6_I0+8:vgprValuB_X6_I0+8+3], v[vgprLocalReadAddrB] offset:18624 // L -> Reg lro=192 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X2_I0+8+0+0:vgprValuB_X2_I0+8+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=3 skipReadsIterA=2 readsPerIterA=1 */
/* dataAtIterB=0 numReadsIterB=3 skipReadsIterB=2 readsPerIterB=3 */

/* iter 3 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:9  */
/* localReadsVacancy: latencyLeft 2 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=8
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X2_I0+0+2+0:vgprValuA_X2_I0+0+2+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:10  */
/* localReadsVacancy: latencyLeft 2 */
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X2_I0+4+2+0:vgprValuB_X2_I0+4+2+0+1], v[vgprValuA_X2_I0+0+2+0:vgprValuA_X2_I0+0+2+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:11  */
/* localReadsVacancy: latencyLeft 2 */
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X2_I0+8+2+0:vgprValuB_X2_I0+8+2+0+1], v[vgprValuA_X2_I0+0+2+0:vgprValuA_X2_I0+0+2+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=3 skipReadsIterA=2 readsPerIterA=1 */
/* dataAtIterB=0 numReadsIterB=3 skipReadsIterB=2 readsPerIterB=3 */

/* iter 4 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:12  */
/* localReadsVacancy: latencyLeft 2 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=4
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+1], v[vgprValuA_X4_I0+0+0+0:vgprValuA_X4_I0+0+0+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:13  */
/* localReadsVacancy: latencyLeft 2 */
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X4_I0+4+0+0:vgprValuB_X4_I0+4+0+0+1], v[vgprValuA_X4_I0+0+0+0:vgprValuA_X4_I0+0+0+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:14  */
/* localReadsVacancy: latencyLeft 2 */
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X4_I0+8+0+0:vgprValuB_X4_I0+8+0+0+1], v[vgprValuA_X4_I0+0+0+0:vgprValuA_X4_I0+0+0+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=1 numReadsIterB=3 skipReadsIterB=1 readsPerIterB=3 */

/* iter 5 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:15  */
/* localReadsVacancy: latencyLeft 2 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=4
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X4_I0+0+2+0:vgprValuB_X4_I0+0+2+0+1], v[vgprValuA_X4_I0+0+2+0:vgprValuA_X4_I0+0+2+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:16  */
/* localReadsVacancy: latencyLeft 2 */
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X4_I0+4+2+0:vgprValuB_X4_I0+4+2+0+1], v[vgprValuA_X4_I0+0+2+0:vgprValuA_X4_I0+0+2+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:17  */
/* schedule remaining localreads for 1LDSB */
/* localReadsVacancy: latencyLeft 2 */
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X4_I0+8+2+0:vgprValuB_X4_I0+8+2+0+1], v[vgprValuA_X4_I0+0+2+0:vgprValuA_X4_I0+0+2+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=1 numReadsIterB=3 skipReadsIterB=1 readsPerIterB=3 */

/* iter 6 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:18  */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+1], v[vgprValuA_X6_I0+0+0+0:vgprValuA_X6_I0+0+0+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:19  */
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X6_I0+4+0+0:vgprValuB_X6_I0+4+0+0+1], v[vgprValuA_X6_I0+0+0+0:vgprValuA_X6_I0+0+0+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:20  */
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X6_I0+8+0+0:vgprValuB_X6_I0+8+0+0+1], v[vgprValuA_X6_I0+0+0+0:vgprValuA_X6_I0+0+0+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=1 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=0 readsPerIterB=3 */

/* iter 7 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:21  */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X6_I0+0+2+0:vgprValuB_X6_I0+0+2+0+1], v[vgprValuA_X6_I0+0+2+0:vgprValuA_X6_I0+0+2+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:22  */
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X6_I0+4+2+0:vgprValuB_X6_I0+4+2+0+1], v[vgprValuA_X6_I0+0+2+0:vgprValuA_X6_I0+0+2+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:23  */
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X6_I0+8+2+0:vgprValuB_X6_I0+8+2+0+1], v[vgprValuA_X6_I0+0+2+0:vgprValuA_X6_I0+0+2+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=1 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=0 readsPerIterB=3 */
label_toPGR1end_OptNLL_0:
/* Stores for OptNLL */
label_Summation_End_OptNLL:
/* endSummation: add vgpr [12...114) to pool */
/* load store sgprs */
.set sgprAddressScaleAlphaVec, 28
.set sgpractivationAlpha, 30
.set sgpractivationBeta, 31
.set sgprActivationType, 32
/* Check if custom structure pointer is null */
s_cmp_eq_u32 s[sgprArgType], 2                     // ArgType == 2 ?
s_cbranch_scc1 label_LoadExternalEpilogueStruct    // branch if ArgType == 2
s_load_dwordx4 s[28:31], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58
s_load_dword s32, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x68
s_branch label_LoadExternalEpilogueStructEnd
label_LoadExternalEpilogueStruct:
s_load_dwordx2 s[28:29], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x90
s_load_dwordx2 s[30:31], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xb8
s_load_dword s32, s[sgprKernArgAddress:sgprKernArgAddress+1], 0xc0
label_LoadExternalEpilogueStructEnd:
.set sgprSrdScaleAlphaVec, 40

/* Mapping of Acc register -> C Vgpr register */

/* Multiply MI out register with Alpha -> C Vgpr register */
/* computeStoreVgprs */
v_lshrrev_b32 v16, 6, v[vgprSerial]                // v16 = v[vgprSerial] / 64
v_lshrrev_b32 v17, 1, v16                          // v17 = v16 / 2
v_mul_lo_u32 v17, 0x10, v17                        // wave coordination offset 1
v_and_b32 v13, 63, v[vgprSerial]                   // v13 = v[vgprSerial] % 64
v_lshrrev_b32 v13, 4, v13                          // v13 = v13 / 16
v_lshlrev_b32 v13, 0x2, v13                        // thread0 * continuous_output
v_add_lshl_u32 v13, v17, v13, 0                    // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v14, v13, s[sgprStrideC1J]            //  offset 1
v_mul_lo_u32 v15, v13, s[sgprStrideD1J]            //  offset 1
v_and_b32 v12, 1, v16                              // v12 = v16 % 2
v_mul_lo_u32 v12, 0x10, v12                        // wave coordination offset 0
v_and_b32 v17, 15, v[vgprSerial]                   // v17 = v[vgprSerial] % 16
v_add_lshl_u32 v12, v17, v12, 0                    // coordination 0 = vwA * (wave_id0 + tid0)
s_mul_i32 s8, 32, s[sgprWorkGroup0]                // wgp0 * MT0
v_add_u32 v12, s8, v12                             // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s8, 96, s[sgprWorkGroup1]                // wgp1 * MT1
v_add_u32 v13, s8, v13                             // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1

/******************************************/
/* Global Write Elements                  */
/******************************************/
s_waitcnt lgkmcnt(0)                               // wait for 20 bytes of kern args.
s_mov_b32 s[sgprSrdScaleAlphaVec+0], s[sgprAddressScaleAlphaVec+0] // init SRD base address (lower)
s_mov_b32 s[sgprSrdScaleAlphaVec+1], s[sgprAddressScaleAlphaVec+1] // init SRD base address (upper) + other fields
s_mov_b32 s[sgprSrdScaleAlphaVec+3], Srd127_96     // Set bits 127_96 in post-loop SRD
s_cmp_eq_u64 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], 0 // s[AddressScaleAlphaVec] == 0 ?
s_cbranch_scc0 label_ScaleAlphaVecAddrValid        // branch if s[AddressScaleAlphaVec] != 0
s_mov_b32 s[sgprSrdScaleAlphaVec+2], 0
s_branch label_ScaleAlphaVecAddrValid_End
label_ScaleAlphaVecAddrValid:
s_mov_b32 s[sgprSrdScaleAlphaVec+2], s[sgprSizeI]
label_ScaleAlphaVecAddrValid_End:

s_mul_i32 s[sgprSrdScaleAlphaVec+2], 0x4, s[sgprSrdScaleAlphaVec+2] // ScaleAlphaVec scaled by BPE

/******************************************/
/* Read vector to LDS                     */
/******************************************/
s_mul_i32 s11, 32, s[sgprWorkGroup0]               // wgp0 * MT0
v_add_u32 v18, s11, v[vgprSerial]                  // coord 0 = wgp0 * MT0 + thread offset
v_lshlrev_b32 v17, 0x2, v18                        // Global scaleAlpha address scaled by BPE
s_mul_i32 s11, 96, s[sgprWorkGroup1]               // wgp1 * MT1
v_add_u32 v18, s11, v[vgprSerial]                  // coord 1 = wgp1 * MT1 + thread offset
buffer_load_dword v16, v17, s[sgprSrdScaleAlphaVec:sgprSrdScaleAlphaVec+3], 0 offen offset:0 // Load ScaleAlphaVec
v_lshlrev_b32 v18, 0x2, v[vgprSerial]              // Local address scaled by BPE
s_barrier                                          // wait for all global loads.
v_cmp_gt_u32 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], s[sgprSrdScaleAlphaVec+2], 0 //  == 0 ?
s_waitcnt vmcnt(0)                                 // wait for global load
v_cndmask_b32 v16, 1, v16, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
ds_write_b32 v18, v16 offset:0                     // store scaleAlpha
.set sgprAddressScaleAlphaVec, UNDEF
.set sgprSrdScaleAlphaVec, UNDEF
s_cmpk_eq_u32 s[sgprActivationType], 5             // activationType == 5
s_cbranch_scc1 label_To_Activation_Relu_VW1        // Branch if true
label_To_Activation_None_VW1:
s_getpc_b64 s[12:13]                               // addr of next instr
s_add_i32 s8, label_Activation_None_VW1, 0x4       // target branch offset
s_add_u32 s12, s12, s8                             // add target branch offset
s_addc_u32 s13, s13, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd
label_To_Activation_Relu_VW1:
s_getpc_b64 s[12:13]                               // addr of next instr
s_add_i32 s8, label_Activation_Relu_VW1, 0x4       // target branch offset
s_add_u32 s12, s12, s8                             // add target branch offset
s_addc_u32 s13, s13, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd
label_ActivationSetPCAddrEnd:
label_GW_B0_E0:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,1,0:vw1); (0,0,2,0:vw1); (0,0,3,0:vw1); (1,0,0,0:vw1); (1,0,1,0:vw1); (1,0,2,0:vw1); (1,0,3,0:vw1); (2,0,0,0:vw1); (2,0,1,0:vw1); (2,0,2,0:vw1); (2,0,3,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
s_mul_i32 s34, 32, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v24, v12, s34
v_lshlrev_b32 v24, 0x2, v24                        // ScaleAlpha address scaled by BPE
s_waitcnt lgkmcnt(0)                               // Wait for LDS write
s_barrier                                          // LDS write barrier
ds_read_b32 v25, v24 offset:0                      // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,1,0,0) */
/* (d1,vc1,d0,vc0)=(1,2,0,0) */
/* (d1,vc1,d0,vc0)=(1,3,0,0) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
/* (d1,vc1,d0,vc0)=(2,1,0,0) */
/* (d1,vc1,d0,vc0)=(2,2,0,0) */
/* (d1,vc1,d0,vc0)=(2,3,0,0) */
v_add_lshl_u32 v22, v15, v12, 0x0                  // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=12, coord0Vgpr=12
v_mov_b32 v[vgprValuC+26], v[vgprValuC+0]          // copy MI out reg to vreg[0]
v_mov_b32 v[vgprValuC+27], v[vgprValuC+1]          // copy MI out reg to vreg[1]
v_mov_b32 v[vgprValuC+28], v[vgprValuC+2]          // copy MI out reg to vreg[2]
v_mov_b32 v[vgprValuC+29], v[vgprValuC+3]          // copy MI out reg to vreg[3]
v_mov_b32 v[vgprValuC+30], v[vgprValuC+4]          // copy MI out reg to vreg[4]
v_mov_b32 v[vgprValuC+31], v[vgprValuC+5]          // copy MI out reg to vreg[5]
v_mov_b32 v[vgprValuC+32], v[vgprValuC+6]          // copy MI out reg to vreg[6]
v_mov_b32 v[vgprValuC+33], v[vgprValuC+7]          // copy MI out reg to vreg[7]
v_mov_b32 v[vgprValuC+34], v[vgprValuC+8]          // copy MI out reg to vreg[8]
v_mov_b32 v[vgprValuC+35], v[vgprValuC+9]          // copy MI out reg to vreg[9]
v_mov_b32 v[vgprValuC+36], v[vgprValuC+10]         // copy MI out reg to vreg[10]
v_mov_b32 v[vgprValuC+37], v[vgprValuC+11]         // copy MI out reg to vreg[11]

/* apply mask, calc new C and issue writes */

s_waitcnt lgkmcnt(0)                               // lgkmcnt(0) = 1 - 1 (scaleAlphaVec) (interleaved)
v_mul_lo_u32 v[vgprValuC+26], v25, v[vgprValuC+26] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v26
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v26, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+26], v[vgprValuC+26], s34, v18 // x= min(127, max(-128, x))
buffer_store_byte v26, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+27], v25, v[vgprValuC+27] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v27
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v27, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+27], v[vgprValuC+27], s34, v18 // x= min(127, max(-128, x))
s_lshl_b32 s34, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v27, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+28], v25, v[vgprValuC+28] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v28
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v28, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+28], v[vgprValuC+28], s34, v18 // x= min(127, max(-128, x))
s_lshl_b32 s34, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v28, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+29], v25, v[vgprValuC+29] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v29
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v29, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+29], v[vgprValuC+29], s34, v18 // x= min(127, max(-128, x))
s_lshl_b32 s34, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v29, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+30], v25, v[vgprValuC+30] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v30
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v30, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+30], v[vgprValuC+30], s34, v18 // x= min(127, max(-128, x))
s_mul_i32 s34, s[sgprStrideD1J], 29                // scale StrideD *= numRows(29) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v30, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+31], v25, v[vgprValuC+31] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v31
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v31, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+31], v[vgprValuC+31], s34, v18 // x= min(127, max(-128, x))
s_lshl_b32 s34, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v31, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+32], v25, v[vgprValuC+32] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v32
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v32, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+32], v[vgprValuC+32], s34, v18 // x= min(127, max(-128, x))
s_lshl_b32 s34, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v32, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+33], v25, v[vgprValuC+33] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v33
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v33, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+33], v[vgprValuC+33], s34, v18 // x= min(127, max(-128, x))
s_lshl_b32 s34, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v33, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+34], v25, v[vgprValuC+34] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v34
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v34, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+34], v[vgprValuC+34], s34, v18 // x= min(127, max(-128, x))
s_mul_i32 s34, s[sgprStrideD1J], 29                // scale StrideD *= numRows(29) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v34, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+35], v25, v[vgprValuC+35] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v35
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v35, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+35], v[vgprValuC+35], s34, v18 // x= min(127, max(-128, x))
s_lshl_b32 s34, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v35, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+36], v25, v[vgprValuC+36] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v36
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v36, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+36], v[vgprValuC+36], s34, v18 // x= min(127, max(-128, x))
s_lshl_b32 s34, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v36, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+37], v25, v[vgprValuC+37] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v37
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v37, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+37], v[vgprValuC+37], s34, v18 // x= min(127, max(-128, x))
s_lshl_b32 s34, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v37, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End                              // jump to end
label_GW_End:

s_endpgm                                           // Kernel End
label_OptNLL_End:
label_GSU_3:

/******************************************/
/* Ord. NoLoadLoop - Begin                */
/******************************************/

/* iter 0 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:0  */
s_waitcnt lgkmcnt(2)                               // wait for prior local read local write old=0, new=2 newLW=0 newLR=2 for iteration == 0
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:1  */
ds_read_b128 v[vgprValuA_X2_I0+0:vgprValuA_X2_I0+0+3], v[vgprLocalReadAddrA] offset:64 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
ds_read_b128 v[vgprValuB_X2_I0+0:vgprValuB_X2_I0+0+3], v[vgprLocalReadAddrB] offset:64 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
s_waitcnt lgkmcnt(2)                               // wait for prior local read local write
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:2  */
ds_read_b128 v[vgprValuB_X2_I0+4:vgprValuB_X2_I0+4+3], v[vgprLocalReadAddrB] offset:9280 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
ds_read_b128 v[vgprValuB_X2_I0+8:vgprValuB_X2_I0+8+3], v[vgprLocalReadAddrB] offset:18496 // L -> Reg lro=64 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=2 iui=0
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X0_I0+8+0+0:vgprValuB_X0_I0+8+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=1 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=-1 numReadsIterB=1 skipReadsIterB=1 readsPerIterB=3 */

/* iter 1 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:3  */
ds_read_b128 v[vgprValuA_X4_I0+0:vgprValuA_X4_I0+0+3], v[vgprLocalReadAddrA] offset:128 // L -> Reg lro=128 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
ds_read_b128 v[vgprValuB_X4_I0+0:vgprValuB_X4_I0+0+3], v[vgprLocalReadAddrB] offset:128 // L -> Reg lro=128 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X0_I0+0+2+0:vgprValuB_X0_I0+0+2+0+1], v[vgprValuA_X0_I0+0+2+0:vgprValuA_X0_I0+0+2+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:4  */
ds_read_b128 v[vgprValuB_X4_I0+4:vgprValuB_X4_I0+4+3], v[vgprLocalReadAddrB] offset:9344 // L -> Reg lro=128 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X0_I0+4+2+0:vgprValuB_X0_I0+4+2+0+1], v[vgprValuA_X0_I0+0+2+0:vgprValuA_X0_I0+0+2+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:5  */
ds_read_b128 v[vgprValuB_X4_I0+8:vgprValuB_X4_I0+8+3], v[vgprLocalReadAddrB] offset:18560 // L -> Reg lro=128 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=4 iui=0
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X0_I0+8+2+0:vgprValuB_X0_I0+8+2+0+1], v[vgprValuA_X0_I0+0+2+0:vgprValuA_X0_I0+0+2+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=-1 numReadsIterA=2 skipReadsIterA=2 readsPerIterA=1 */
/* dataAtIterB=-1 numReadsIterB=2 skipReadsIterB=2 readsPerIterB=3 */

/* iter 2 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:6  */
ds_read_b128 v[vgprValuA_X6_I0+0:vgprValuA_X6_I0+0+3], v[vgprLocalReadAddrA] offset:192 // L -> Reg lro=192 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
ds_read_b128 v[vgprValuB_X6_I0+0:vgprValuB_X6_I0+0+3], v[vgprLocalReadAddrB] offset:192 // L -> Reg lro=192 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
s_waitcnt lgkmcnt(6)                               // wait for prior local read local write old=0, new=6 newLW=0 newLR=6
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X2_I0+0+0+0:vgprValuB_X2_I0+0+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:7  */
ds_read_b128 v[vgprValuB_X6_I0+4:vgprValuB_X6_I0+4+3], v[vgprLocalReadAddrB] offset:9408 // L -> Reg lro=192 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X2_I0+4+0+0:vgprValuB_X2_I0+4+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:8  */
ds_read_b128 v[vgprValuB_X6_I0+8:vgprValuB_X6_I0+8+3], v[vgprLocalReadAddrB] offset:18624 // L -> Reg lro=192 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=6 iui=0
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X2_I0+8+0+0:vgprValuB_X2_I0+8+0+0+1], v[vgprValuA_X2_I0+0+0+0:vgprValuA_X2_I0+0+0+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=3 skipReadsIterA=2 readsPerIterA=1 */
/* dataAtIterB=0 numReadsIterB=3 skipReadsIterB=2 readsPerIterB=3 */

/* iter 3 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:9  */
/* localReadsVacancy: latencyLeft 2 */
s_waitcnt lgkmcnt(8)                               // wait for prior local read local write old=0, new=8 newLW=0 newLR=8
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X2_I0+0+2+0:vgprValuB_X2_I0+0+2+0+1], v[vgprValuA_X2_I0+0+2+0:vgprValuA_X2_I0+0+2+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:10  */
/* localReadsVacancy: latencyLeft 2 */
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X2_I0+4+2+0:vgprValuB_X2_I0+4+2+0+1], v[vgprValuA_X2_I0+0+2+0:vgprValuA_X2_I0+0+2+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:11  */
/* localReadsVacancy: latencyLeft 2 */
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X2_I0+8+2+0:vgprValuB_X2_I0+8+2+0+1], v[vgprValuA_X2_I0+0+2+0:vgprValuA_X2_I0+0+2+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=0 numReadsIterA=3 skipReadsIterA=2 readsPerIterA=1 */
/* dataAtIterB=0 numReadsIterB=3 skipReadsIterB=2 readsPerIterB=3 */

/* iter 4 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:12  */
/* localReadsVacancy: latencyLeft 2 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=4
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X4_I0+0+0+0:vgprValuB_X4_I0+0+0+0+1], v[vgprValuA_X4_I0+0+0+0:vgprValuA_X4_I0+0+0+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:13  */
/* localReadsVacancy: latencyLeft 2 */
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X4_I0+4+0+0:vgprValuB_X4_I0+4+0+0+1], v[vgprValuA_X4_I0+0+0+0:vgprValuA_X4_I0+0+0+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:14  */
/* localReadsVacancy: latencyLeft 2 */
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X4_I0+8+0+0:vgprValuB_X4_I0+8+0+0+1], v[vgprValuA_X4_I0+0+0+0:vgprValuA_X4_I0+0+0+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=1 numReadsIterB=3 skipReadsIterB=1 readsPerIterB=3 */

/* iter 5 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:15  */
/* localReadsVacancy: latencyLeft 2 */
s_waitcnt lgkmcnt(4)                               // wait for prior local read local write old=0, new=4 newLW=0 newLR=4
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X4_I0+0+2+0:vgprValuB_X4_I0+0+2+0+1], v[vgprValuA_X4_I0+0+2+0:vgprValuA_X4_I0+0+2+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:16  */
/* localReadsVacancy: latencyLeft 2 */
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X4_I0+4+2+0:vgprValuB_X4_I0+4+2+0+1], v[vgprValuA_X4_I0+0+2+0:vgprValuA_X4_I0+0+2+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:17  */
/* schedule remaining localreads for 1LDSB */
/* localReadsVacancy: latencyLeft 2 */
/* 1 LDS buffer: read-sync-write */
s_waitcnt lgkmcnt(0)
s_barrier
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X4_I0+8+2+0:vgprValuB_X4_I0+8+2+0+1], v[vgprValuA_X4_I0+0+2+0:vgprValuA_X4_I0+0+2+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=1 numReadsIterA=3 skipReadsIterA=1 readsPerIterA=1 */
/* dataAtIterB=1 numReadsIterB=3 skipReadsIterB=1 readsPerIterB=3 */

/* iter 6 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:18  */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X6_I0+0+0+0:vgprValuB_X6_I0+0+0+0+1], v[vgprValuA_X6_I0+0+0+0:vgprValuA_X6_I0+0+0+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:19  */
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X6_I0+4+0+0:vgprValuB_X6_I0+4+0+0+1], v[vgprValuA_X6_I0+0+0+0:vgprValuA_X6_I0+0+0+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:20  */
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X6_I0+8+0+0:vgprValuB_X6_I0+8+0+0+1], v[vgprValuA_X6_I0+0+0+0:vgprValuA_X6_I0+0+0+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=1 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=0 readsPerIterB=3 */

/* iter 7 (last unrolled loop) */
/*  grEndMfmaIndex:0, lwStartMfmaIndex:18, lwEndMfmaIndex:18  */
/*  numMfmaForLR:2, syncPlrMfmaIndex:21  */
/*  mfmaIndex:21  */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0, new=0 newLW=0 newLR=0
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X6_I0+0+2+0:vgprValuB_X6_I0+0+2+0+1], v[vgprValuA_X6_I0+0+2+0:vgprValuA_X6_I0+0+2+0+1], v[0:3] // left value = v[0+0:3+0]
/*  mfmaIndex:22  */
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X6_I0+4+2+0:vgprValuB_X6_I0+4+2+0+1], v[vgprValuA_X6_I0+0+2+0:vgprValuA_X6_I0+0+2+0+1], v[4:7] // left value = v[4+0:7+0]
/*  mfmaIndex:23  */
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X6_I0+8+2+0:vgprValuB_X6_I0+8+2+0+1], v[vgprValuA_X6_I0+0+2+0:vgprValuA_X6_I0+0+2+0+1], v[8:11] // left value = v[8+0:11+0]
/* numPrefetchIter=0 */
/* dataAtIterA=2 numReadsIterA=3 skipReadsIterA=0 readsPerIterA=1 */
/* dataAtIterB=2 numReadsIterB=3 skipReadsIterB=0 readsPerIterB=3 */
label_toPGR1end_OrdNLL_0:
label_PrefetchGlobalLastIterEnd:

/******************************************/
/* Tail Loop                              */
/******************************************/

/* Tail: add ValuA/B vgpr buffer [12...88) to pool */

/* local write reset offsets a */

/* local write reset offsets b */

// numIterL = LOCAL_SPLITU * min(sizeL % LOCAL_DEPTHU, DEPTHU / LOCAL_SPLITU)
s_and_b32 s[sgprLoopCounterL], 255, s[sgprSizesSum+0] // s[sgprLoopCounterL] = s[sgprSizesSum+0] % 256
s_and_b32 s28, s[sgprGSU], 0x8000                  // SCC = (GSUC == 1) ?
s_cbranch_scc1 label_GSUC_TL                       // branch if GSUC == 1
s_cmp_lg_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx == numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], 0x0                // numIter=0 if gsuSimIdx != numIterPerWgRemainder
s_branch label_GSUC_TL_End
label_GSUC_TL:
s_lshr_b32 s29, s[sgprSizesSum], 8                 // s29 = s[sgprSizesSum] / 256
s_and_b32 s30, s[sgprGSU], 0x3fff                  // Restore GSU
v_cvt_f32_u32 v12, s30                             // s28 = s29 / s30
v_rcp_iflag_f32 v12, v12                           // s28 = s29 / s30
v_cvt_f32_u32 v13, s29                             // s28 = s29 / s30
v_mul_f32 v12, v12, v13                            // s28 = s29 / s30
v_cvt_u32_f32 v12, v12                             // s28 = s29 / s30
v_mul_u32_u24 v13, v12, s30                        // s28 = s29 / s30
v_sub_u32 v13, s29, v13                            // s28 = s29 / s30
v_cmpx_eq_u32 exec, v13, s30                       // s28 = s29 / s30
v_add_u32 v12, 1, v12                              // s28 = s29 / s30
v_mov_b32 v13, 0                                   // s[sgprGSUSumIdx+1] = s29 % s30
s_mov_b64 exec, -1                                 // s28 = s29 / s30
v_readfirstlane_b32 s28, v12                       // quotient
v_readfirstlane_b32 s[sgprGSUSumIdx+1], v13        // remainder
s_sub_u32 s29, s30, 1                              // GSU-1
s_cmp_eq_u32 s28, 0                                // quotient == 0
s_cselect_b32 s28, s[sgprGSUSumIdx+1], s29         // lastWg = (quotient==0) ? numIterPerWgRemainder : GSU-1
s_cmp_lg_u32 s[sgprGSUSumIdx], s28                 // gsuSumIdx == lastWg
s_cmov_b32 s[sgprLoopCounterL], 0x0                // numIter=0 if gsuSumIdx != lastWg
label_GSUC_TL_End:
s_cmp_eq_u32 s[sgprLoopCounterL], 0x0              // numIterL == 0
s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment
s_cbranch_scc1 label_SkipTailLoopL                 // skip to end of tail loop b/c numIter==0

/* remove stagger offsets for tail loop */
s_sub_i32 s28, 3, s[sgprStaggerUIter]
s_mul_hi_i32 s29, s28, s[sgprGlobalReadIncsA+0]    // start offset S in bytes
s_mul_i32 s28, s28, s[sgprGlobalReadIncsA+0]       // start offset S in bytes
s_sub_u32 s28, s28, s[sgprWrapUA]                  // S - WrapU
s_subb_u32 s29, s29, s[sgprWrapUA+1]               // S - WrapU
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s28        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], s29       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s28 // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s29 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_sub_i32 s28, 3, s[sgprStaggerUIter]
s_mul_hi_i32 s29, s28, s[sgprGlobalReadIncsB+0]    // start offset S in bytes
s_mul_i32 s28, s28, s[sgprGlobalReadIncsB+0]       // start offset S in bytes
s_sub_u32 s28, s28, s[sgprWrapUB]                  // S - WrapU
s_subb_u32 s29, s29, s[sgprWrapUB+1]               // S - WrapU
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s28        // gra SRD += inc(lower)
s_addc_u32 s[sgprSrdB+1], s[sgprSrdB+1], s29       // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s28 // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s29 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32

/* Update M0 for DTLDS */

/* global read A */
/* g2l=0, load component 0 */
buffer_load_ubyte_d16 v[vgprG2LA+0+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
buffer_load_ubyte_d16 v12, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:1 // load one buffer value
/* g2l=0, load component 2 */
buffer_load_ubyte_d16_hi v13, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:2 // load one buffer value
/* g2l=0, load component 3 */
buffer_load_ubyte_d16_hi v14, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:3 // load one buffer value
/* g2l=0, load component 4 */
buffer_load_ubyte_d16 v[vgprG2LA+0+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:4 // load one buffer value
/* g2l=0, load component 5 */
buffer_load_ubyte_d16 v16, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:5 // load one buffer value
/* g2l=0, load component 6 */
buffer_load_ubyte_d16_hi v17, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:6 // load one buffer value
/* g2l=0, load component 7 */
buffer_load_ubyte_d16_hi v18, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:7 // load one buffer value
/* g2l=0, load component 8 */
buffer_load_ubyte_d16 v[vgprG2LA+0+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:8 // load one buffer value
/* g2l=0, load component 9 */
buffer_load_ubyte_d16 v20, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:9 // load one buffer value
/* g2l=0, load component 10 */
buffer_load_ubyte_d16_hi v21, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:10 // load one buffer value
/* g2l=0, load component 11 */
buffer_load_ubyte_d16_hi v22, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:11 // load one buffer value
/* g2l=0, load component 12 */
buffer_load_ubyte_d16 v[vgprG2LA+0+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:12 // load one buffer value
/* g2l=0, load component 13 */
buffer_load_ubyte_d16 v24, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:13 // load one buffer value
/* g2l=0, load component 14 */
buffer_load_ubyte_d16_hi v25, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:14 // load one buffer value
/* g2l=0, load component 15 */
buffer_load_ubyte_d16_hi v26, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0 offen offset:15 // load one buffer value
s_waitcnt vmcnt(14)
v_lshlrev_b32 v12, 0x8, v12                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+0], v[vgprG2LA+0+0], v12     // pack a sub 8-bit with dest
s_waitcnt vmcnt(13)
v_or_b32 v[vgprG2LA+0+0], v[vgprG2LA+0+0], v13     // pack a sub 8-bit with dest
s_waitcnt vmcnt(12)
v_lshlrev_b32 v14, 0x8, v14                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+0], v[vgprG2LA+0+0], v14     // pack a sub 8-bit with dest
s_waitcnt vmcnt(10)
v_lshlrev_b32 v16, 0x8, v16                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+1], v[vgprG2LA+0+1], v16     // pack a sub 8-bit with dest
s_waitcnt vmcnt(9)
v_or_b32 v[vgprG2LA+0+1], v[vgprG2LA+0+1], v17     // pack a sub 8-bit with dest
s_waitcnt vmcnt(8)
v_lshlrev_b32 v18, 0x8, v18                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+1], v[vgprG2LA+0+1], v18     // pack a sub 8-bit with dest
s_waitcnt vmcnt(6)
v_lshlrev_b32 v20, 0x8, v20                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+2], v[vgprG2LA+0+2], v20     // pack a sub 8-bit with dest
s_waitcnt vmcnt(5)
v_or_b32 v[vgprG2LA+0+2], v[vgprG2LA+0+2], v21     // pack a sub 8-bit with dest
s_waitcnt vmcnt(4)
v_lshlrev_b32 v22, 0x8, v22                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+2], v[vgprG2LA+0+2], v22     // pack a sub 8-bit with dest
s_waitcnt vmcnt(2)
v_lshlrev_b32 v24, 0x8, v24                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+3], v[vgprG2LA+0+3], v24     // pack a sub 8-bit with dest
s_waitcnt vmcnt(1)
v_or_b32 v[vgprG2LA+0+3], v[vgprG2LA+0+3], v25     // pack a sub 8-bit with dest
s_waitcnt vmcnt(0)
v_lshlrev_b32 v26, 0x8, v26                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+0+3], v[vgprG2LA+0+3], v26     // pack a sub 8-bit with dest
/* g2l=4, load component 0 */
buffer_load_ubyte_d16 v[vgprG2LA+4+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:0 // load one buffer value
/* g2l=4, load component 1 */
buffer_load_ubyte_d16 v12, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:1 // load one buffer value
/* g2l=4, load component 2 */
buffer_load_ubyte_d16_hi v13, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:2 // load one buffer value
/* g2l=4, load component 3 */
buffer_load_ubyte_d16_hi v14, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:3 // load one buffer value
/* g2l=4, load component 4 */
buffer_load_ubyte_d16 v[vgprG2LA+4+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:4 // load one buffer value
/* g2l=4, load component 5 */
buffer_load_ubyte_d16 v16, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:5 // load one buffer value
/* g2l=4, load component 6 */
buffer_load_ubyte_d16_hi v17, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:6 // load one buffer value
/* g2l=4, load component 7 */
buffer_load_ubyte_d16_hi v18, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:7 // load one buffer value
/* g2l=4, load component 8 */
buffer_load_ubyte_d16 v[vgprG2LA+4+2], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:8 // load one buffer value
/* g2l=4, load component 9 */
buffer_load_ubyte_d16 v20, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:9 // load one buffer value
/* g2l=4, load component 10 */
buffer_load_ubyte_d16_hi v21, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:10 // load one buffer value
/* g2l=4, load component 11 */
buffer_load_ubyte_d16_hi v22, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:11 // load one buffer value
/* g2l=4, load component 12 */
buffer_load_ubyte_d16 v[vgprG2LA+4+3], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:12 // load one buffer value
/* g2l=4, load component 13 */
buffer_load_ubyte_d16 v24, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:13 // load one buffer value
/* g2l=4, load component 14 */
buffer_load_ubyte_d16_hi v25, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:14 // load one buffer value
/* g2l=4, load component 15 */
buffer_load_ubyte_d16_hi v26, v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0] offen offset:15 // load one buffer value
s_waitcnt vmcnt(14)
v_lshlrev_b32 v12, 0x8, v12                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+4+0], v[vgprG2LA+4+0], v12     // pack a sub 8-bit with dest
s_waitcnt vmcnt(13)
v_or_b32 v[vgprG2LA+4+0], v[vgprG2LA+4+0], v13     // pack a sub 8-bit with dest
s_waitcnt vmcnt(12)
v_lshlrev_b32 v14, 0x8, v14                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+4+0], v[vgprG2LA+4+0], v14     // pack a sub 8-bit with dest
s_waitcnt vmcnt(10)
v_lshlrev_b32 v16, 0x8, v16                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+4+1], v[vgprG2LA+4+1], v16     // pack a sub 8-bit with dest
s_waitcnt vmcnt(9)
v_or_b32 v[vgprG2LA+4+1], v[vgprG2LA+4+1], v17     // pack a sub 8-bit with dest
s_waitcnt vmcnt(8)
v_lshlrev_b32 v18, 0x8, v18                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+4+1], v[vgprG2LA+4+1], v18     // pack a sub 8-bit with dest
s_waitcnt vmcnt(6)
v_lshlrev_b32 v20, 0x8, v20                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+4+2], v[vgprG2LA+4+2], v20     // pack a sub 8-bit with dest
s_waitcnt vmcnt(5)
v_or_b32 v[vgprG2LA+4+2], v[vgprG2LA+4+2], v21     // pack a sub 8-bit with dest
s_waitcnt vmcnt(4)
v_lshlrev_b32 v22, 0x8, v22                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+4+2], v[vgprG2LA+4+2], v22     // pack a sub 8-bit with dest
s_waitcnt vmcnt(2)
v_lshlrev_b32 v24, 0x8, v24                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+4+3], v[vgprG2LA+4+3], v24     // pack a sub 8-bit with dest
s_waitcnt vmcnt(1)
v_or_b32 v[vgprG2LA+4+3], v[vgprG2LA+4+3], v25     // pack a sub 8-bit with dest
s_waitcnt vmcnt(0)
v_lshlrev_b32 v26, 0x8, v26                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LA+4+3], v[vgprG2LA+4+3], v26     // pack a sub 8-bit with dest

/* Update M0 for DTLDS */

/* global read B */
/* g2l=0, load component 0 */
buffer_load_ubyte_d16 v[vgprG2LB+0+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
buffer_load_ubyte_d16 v12, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:1 // load one buffer value
/* g2l=0, load component 2 */
buffer_load_ubyte_d16_hi v13, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:2 // load one buffer value
/* g2l=0, load component 3 */
buffer_load_ubyte_d16_hi v14, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:3 // load one buffer value
/* g2l=0, load component 4 */
buffer_load_ubyte_d16 v[vgprG2LB+0+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:4 // load one buffer value
/* g2l=0, load component 5 */
buffer_load_ubyte_d16 v16, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:5 // load one buffer value
/* g2l=0, load component 6 */
buffer_load_ubyte_d16_hi v17, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:6 // load one buffer value
/* g2l=0, load component 7 */
buffer_load_ubyte_d16_hi v18, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:7 // load one buffer value
/* g2l=0, load component 8 */
buffer_load_ubyte_d16 v[vgprG2LB+0+2], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:8 // load one buffer value
/* g2l=0, load component 9 */
buffer_load_ubyte_d16 v20, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:9 // load one buffer value
/* g2l=0, load component 10 */
buffer_load_ubyte_d16_hi v21, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:10 // load one buffer value
/* g2l=0, load component 11 */
buffer_load_ubyte_d16_hi v22, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:11 // load one buffer value
/* g2l=0, load component 12 */
buffer_load_ubyte_d16 v[vgprG2LB+0+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:12 // load one buffer value
/* g2l=0, load component 13 */
buffer_load_ubyte_d16 v24, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:13 // load one buffer value
/* g2l=0, load component 14 */
buffer_load_ubyte_d16_hi v25, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:14 // load one buffer value
/* g2l=0, load component 15 */
buffer_load_ubyte_d16_hi v26, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0 offen offset:15 // load one buffer value
s_waitcnt vmcnt(14)
v_lshlrev_b32 v12, 0x8, v12                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+0], v[vgprG2LB+0+0], v12     // pack a sub 8-bit with dest
s_waitcnt vmcnt(13)
v_or_b32 v[vgprG2LB+0+0], v[vgprG2LB+0+0], v13     // pack a sub 8-bit with dest
s_waitcnt vmcnt(12)
v_lshlrev_b32 v14, 0x8, v14                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+0], v[vgprG2LB+0+0], v14     // pack a sub 8-bit with dest
s_waitcnt vmcnt(10)
v_lshlrev_b32 v16, 0x8, v16                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+1], v[vgprG2LB+0+1], v16     // pack a sub 8-bit with dest
s_waitcnt vmcnt(9)
v_or_b32 v[vgprG2LB+0+1], v[vgprG2LB+0+1], v17     // pack a sub 8-bit with dest
s_waitcnt vmcnt(8)
v_lshlrev_b32 v18, 0x8, v18                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+1], v[vgprG2LB+0+1], v18     // pack a sub 8-bit with dest
s_waitcnt vmcnt(6)
v_lshlrev_b32 v20, 0x8, v20                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+2], v[vgprG2LB+0+2], v20     // pack a sub 8-bit with dest
s_waitcnt vmcnt(5)
v_or_b32 v[vgprG2LB+0+2], v[vgprG2LB+0+2], v21     // pack a sub 8-bit with dest
s_waitcnt vmcnt(4)
v_lshlrev_b32 v22, 0x8, v22                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+2], v[vgprG2LB+0+2], v22     // pack a sub 8-bit with dest
s_waitcnt vmcnt(2)
v_lshlrev_b32 v24, 0x8, v24                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+3], v[vgprG2LB+0+3], v24     // pack a sub 8-bit with dest
s_waitcnt vmcnt(1)
v_or_b32 v[vgprG2LB+0+3], v[vgprG2LB+0+3], v25     // pack a sub 8-bit with dest
s_waitcnt vmcnt(0)
v_lshlrev_b32 v26, 0x8, v26                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+0+3], v[vgprG2LB+0+3], v26     // pack a sub 8-bit with dest
/* g2l=4, load component 0 */
buffer_load_ubyte_d16 v[vgprG2LB+4+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:0 // load one buffer value
/* g2l=4, load component 1 */
buffer_load_ubyte_d16 v12, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:1 // load one buffer value
/* g2l=4, load component 2 */
buffer_load_ubyte_d16_hi v13, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:2 // load one buffer value
/* g2l=4, load component 3 */
buffer_load_ubyte_d16_hi v14, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:3 // load one buffer value
/* g2l=4, load component 4 */
buffer_load_ubyte_d16 v[vgprG2LB+4+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:4 // load one buffer value
/* g2l=4, load component 5 */
buffer_load_ubyte_d16 v16, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:5 // load one buffer value
/* g2l=4, load component 6 */
buffer_load_ubyte_d16_hi v17, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:6 // load one buffer value
/* g2l=4, load component 7 */
buffer_load_ubyte_d16_hi v18, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:7 // load one buffer value
/* g2l=4, load component 8 */
buffer_load_ubyte_d16 v[vgprG2LB+4+2], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:8 // load one buffer value
/* g2l=4, load component 9 */
buffer_load_ubyte_d16 v20, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:9 // load one buffer value
/* g2l=4, load component 10 */
buffer_load_ubyte_d16_hi v21, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:10 // load one buffer value
/* g2l=4, load component 11 */
buffer_load_ubyte_d16_hi v22, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:11 // load one buffer value
/* g2l=4, load component 12 */
buffer_load_ubyte_d16 v[vgprG2LB+4+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:12 // load one buffer value
/* g2l=4, load component 13 */
buffer_load_ubyte_d16 v24, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:13 // load one buffer value
/* g2l=4, load component 14 */
buffer_load_ubyte_d16_hi v25, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:14 // load one buffer value
/* g2l=4, load component 15 */
buffer_load_ubyte_d16_hi v26, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0] offen offset:15 // load one buffer value
s_waitcnt vmcnt(14)
v_lshlrev_b32 v12, 0x8, v12                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+4+0], v[vgprG2LB+4+0], v12     // pack a sub 8-bit with dest
s_waitcnt vmcnt(13)
v_or_b32 v[vgprG2LB+4+0], v[vgprG2LB+4+0], v13     // pack a sub 8-bit with dest
s_waitcnt vmcnt(12)
v_lshlrev_b32 v14, 0x8, v14                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+4+0], v[vgprG2LB+4+0], v14     // pack a sub 8-bit with dest
s_waitcnt vmcnt(10)
v_lshlrev_b32 v16, 0x8, v16                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+4+1], v[vgprG2LB+4+1], v16     // pack a sub 8-bit with dest
s_waitcnt vmcnt(9)
v_or_b32 v[vgprG2LB+4+1], v[vgprG2LB+4+1], v17     // pack a sub 8-bit with dest
s_waitcnt vmcnt(8)
v_lshlrev_b32 v18, 0x8, v18                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+4+1], v[vgprG2LB+4+1], v18     // pack a sub 8-bit with dest
s_waitcnt vmcnt(6)
v_lshlrev_b32 v20, 0x8, v20                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+4+2], v[vgprG2LB+4+2], v20     // pack a sub 8-bit with dest
s_waitcnt vmcnt(5)
v_or_b32 v[vgprG2LB+4+2], v[vgprG2LB+4+2], v21     // pack a sub 8-bit with dest
s_waitcnt vmcnt(4)
v_lshlrev_b32 v22, 0x8, v22                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+4+2], v[vgprG2LB+4+2], v22     // pack a sub 8-bit with dest
s_waitcnt vmcnt(2)
v_lshlrev_b32 v24, 0x8, v24                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+4+3], v[vgprG2LB+4+3], v24     // pack a sub 8-bit with dest
s_waitcnt vmcnt(1)
v_or_b32 v[vgprG2LB+4+3], v[vgprG2LB+4+3], v25     // pack a sub 8-bit with dest
s_waitcnt vmcnt(0)
v_lshlrev_b32 v26, 0x8, v26                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+4+3], v[vgprG2LB+4+3], v26     // pack a sub 8-bit with dest
/* g2l=8, load component 0 */
buffer_load_ubyte_d16 v[vgprG2LB+8+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+1] offen offset:0 // load one buffer value
/* g2l=8, load component 1 */
buffer_load_ubyte_d16 v12, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+1] offen offset:1 // load one buffer value
/* g2l=8, load component 2 */
buffer_load_ubyte_d16_hi v13, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+1] offen offset:2 // load one buffer value
/* g2l=8, load component 3 */
buffer_load_ubyte_d16_hi v14, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+1] offen offset:3 // load one buffer value
/* g2l=8, load component 4 */
buffer_load_ubyte_d16 v[vgprG2LB+8+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+1] offen offset:4 // load one buffer value
/* g2l=8, load component 5 */
buffer_load_ubyte_d16 v16, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+1] offen offset:5 // load one buffer value
/* g2l=8, load component 6 */
buffer_load_ubyte_d16_hi v17, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+1] offen offset:6 // load one buffer value
/* g2l=8, load component 7 */
buffer_load_ubyte_d16_hi v18, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+1] offen offset:7 // load one buffer value
/* g2l=8, load component 8 */
buffer_load_ubyte_d16 v[vgprG2LB+8+2], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+1] offen offset:8 // load one buffer value
/* g2l=8, load component 9 */
buffer_load_ubyte_d16 v20, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+1] offen offset:9 // load one buffer value
/* g2l=8, load component 10 */
buffer_load_ubyte_d16_hi v21, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+1] offen offset:10 // load one buffer value
/* g2l=8, load component 11 */
buffer_load_ubyte_d16_hi v22, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+1] offen offset:11 // load one buffer value
/* g2l=8, load component 12 */
buffer_load_ubyte_d16 v[vgprG2LB+8+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+1] offen offset:12 // load one buffer value
/* g2l=8, load component 13 */
buffer_load_ubyte_d16 v24, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+1] offen offset:13 // load one buffer value
/* g2l=8, load component 14 */
buffer_load_ubyte_d16_hi v25, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+1] offen offset:14 // load one buffer value
/* g2l=8, load component 15 */
buffer_load_ubyte_d16_hi v26, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+1] offen offset:15 // load one buffer value
s_waitcnt vmcnt(14)
v_lshlrev_b32 v12, 0x8, v12                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+8+0], v[vgprG2LB+8+0], v12     // pack a sub 8-bit with dest
s_waitcnt vmcnt(13)
v_or_b32 v[vgprG2LB+8+0], v[vgprG2LB+8+0], v13     // pack a sub 8-bit with dest
s_waitcnt vmcnt(12)
v_lshlrev_b32 v14, 0x8, v14                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+8+0], v[vgprG2LB+8+0], v14     // pack a sub 8-bit with dest
s_waitcnt vmcnt(10)
v_lshlrev_b32 v16, 0x8, v16                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+8+1], v[vgprG2LB+8+1], v16     // pack a sub 8-bit with dest
s_waitcnt vmcnt(9)
v_or_b32 v[vgprG2LB+8+1], v[vgprG2LB+8+1], v17     // pack a sub 8-bit with dest
s_waitcnt vmcnt(8)
v_lshlrev_b32 v18, 0x8, v18                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+8+1], v[vgprG2LB+8+1], v18     // pack a sub 8-bit with dest
s_waitcnt vmcnt(6)
v_lshlrev_b32 v20, 0x8, v20                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+8+2], v[vgprG2LB+8+2], v20     // pack a sub 8-bit with dest
s_waitcnt vmcnt(5)
v_or_b32 v[vgprG2LB+8+2], v[vgprG2LB+8+2], v21     // pack a sub 8-bit with dest
s_waitcnt vmcnt(4)
v_lshlrev_b32 v22, 0x8, v22                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+8+2], v[vgprG2LB+8+2], v22     // pack a sub 8-bit with dest
s_waitcnt vmcnt(2)
v_lshlrev_b32 v24, 0x8, v24                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+8+3], v[vgprG2LB+8+3], v24     // pack a sub 8-bit with dest
s_waitcnt vmcnt(1)
v_or_b32 v[vgprG2LB+8+3], v[vgprG2LB+8+3], v25     // pack a sub 8-bit with dest
s_waitcnt vmcnt(0)
v_lshlrev_b32 v26, 0x8, v26                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+8+3], v[vgprG2LB+8+3], v26     // pack a sub 8-bit with dest
/* g2l=12, load component 0 */
buffer_load_ubyte_d16 v[vgprG2LB+12+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+2] offen offset:0 // load one buffer value
/* g2l=12, load component 1 */
buffer_load_ubyte_d16 v12, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+2] offen offset:1 // load one buffer value
/* g2l=12, load component 2 */
buffer_load_ubyte_d16_hi v13, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+2] offen offset:2 // load one buffer value
/* g2l=12, load component 3 */
buffer_load_ubyte_d16_hi v14, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+2] offen offset:3 // load one buffer value
/* g2l=12, load component 4 */
buffer_load_ubyte_d16 v[vgprG2LB+12+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+2] offen offset:4 // load one buffer value
/* g2l=12, load component 5 */
buffer_load_ubyte_d16 v16, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+2] offen offset:5 // load one buffer value
/* g2l=12, load component 6 */
buffer_load_ubyte_d16_hi v17, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+2] offen offset:6 // load one buffer value
/* g2l=12, load component 7 */
buffer_load_ubyte_d16_hi v18, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+2] offen offset:7 // load one buffer value
/* g2l=12, load component 8 */
buffer_load_ubyte_d16 v[vgprG2LB+12+2], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+2] offen offset:8 // load one buffer value
/* g2l=12, load component 9 */
buffer_load_ubyte_d16 v20, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+2] offen offset:9 // load one buffer value
/* g2l=12, load component 10 */
buffer_load_ubyte_d16_hi v21, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+2] offen offset:10 // load one buffer value
/* g2l=12, load component 11 */
buffer_load_ubyte_d16_hi v22, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+2] offen offset:11 // load one buffer value
/* g2l=12, load component 12 */
buffer_load_ubyte_d16 v[vgprG2LB+12+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+2] offen offset:12 // load one buffer value
/* g2l=12, load component 13 */
buffer_load_ubyte_d16 v24, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+2] offen offset:13 // load one buffer value
/* g2l=12, load component 14 */
buffer_load_ubyte_d16_hi v25, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+2] offen offset:14 // load one buffer value
/* g2l=12, load component 15 */
buffer_load_ubyte_d16_hi v26, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+2] offen offset:15 // load one buffer value
s_waitcnt vmcnt(14)
v_lshlrev_b32 v12, 0x8, v12                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+12+0], v[vgprG2LB+12+0], v12   // pack a sub 8-bit with dest
s_waitcnt vmcnt(13)
v_or_b32 v[vgprG2LB+12+0], v[vgprG2LB+12+0], v13   // pack a sub 8-bit with dest
s_waitcnt vmcnt(12)
v_lshlrev_b32 v14, 0x8, v14                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+12+0], v[vgprG2LB+12+0], v14   // pack a sub 8-bit with dest
s_waitcnt vmcnt(10)
v_lshlrev_b32 v16, 0x8, v16                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+12+1], v[vgprG2LB+12+1], v16   // pack a sub 8-bit with dest
s_waitcnt vmcnt(9)
v_or_b32 v[vgprG2LB+12+1], v[vgprG2LB+12+1], v17   // pack a sub 8-bit with dest
s_waitcnt vmcnt(8)
v_lshlrev_b32 v18, 0x8, v18                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+12+1], v[vgprG2LB+12+1], v18   // pack a sub 8-bit with dest
s_waitcnt vmcnt(6)
v_lshlrev_b32 v20, 0x8, v20                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+12+2], v[vgprG2LB+12+2], v20   // pack a sub 8-bit with dest
s_waitcnt vmcnt(5)
v_or_b32 v[vgprG2LB+12+2], v[vgprG2LB+12+2], v21   // pack a sub 8-bit with dest
s_waitcnt vmcnt(4)
v_lshlrev_b32 v22, 0x8, v22                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+12+2], v[vgprG2LB+12+2], v22   // pack a sub 8-bit with dest
s_waitcnt vmcnt(2)
v_lshlrev_b32 v24, 0x8, v24                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+12+3], v[vgprG2LB+12+3], v24   // pack a sub 8-bit with dest
s_waitcnt vmcnt(1)
v_or_b32 v[vgprG2LB+12+3], v[vgprG2LB+12+3], v25   // pack a sub 8-bit with dest
s_waitcnt vmcnt(0)
v_lshlrev_b32 v26, 0x8, v26                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+12+3], v[vgprG2LB+12+3], v26   // pack a sub 8-bit with dest
/* g2l=16, load component 0 */
buffer_load_ubyte_d16 v[vgprG2LB+16+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+3] offen offset:0 // load one buffer value
/* g2l=16, load component 1 */
buffer_load_ubyte_d16 v12, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+3] offen offset:1 // load one buffer value
/* g2l=16, load component 2 */
buffer_load_ubyte_d16_hi v13, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+3] offen offset:2 // load one buffer value
/* g2l=16, load component 3 */
buffer_load_ubyte_d16_hi v14, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+3] offen offset:3 // load one buffer value
/* g2l=16, load component 4 */
buffer_load_ubyte_d16 v[vgprG2LB+16+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+3] offen offset:4 // load one buffer value
/* g2l=16, load component 5 */
buffer_load_ubyte_d16 v16, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+3] offen offset:5 // load one buffer value
/* g2l=16, load component 6 */
buffer_load_ubyte_d16_hi v17, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+3] offen offset:6 // load one buffer value
/* g2l=16, load component 7 */
buffer_load_ubyte_d16_hi v18, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+3] offen offset:7 // load one buffer value
/* g2l=16, load component 8 */
buffer_load_ubyte_d16 v[vgprG2LB+16+2], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+3] offen offset:8 // load one buffer value
/* g2l=16, load component 9 */
buffer_load_ubyte_d16 v20, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+3] offen offset:9 // load one buffer value
/* g2l=16, load component 10 */
buffer_load_ubyte_d16_hi v21, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+3] offen offset:10 // load one buffer value
/* g2l=16, load component 11 */
buffer_load_ubyte_d16_hi v22, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+3] offen offset:11 // load one buffer value
/* g2l=16, load component 12 */
buffer_load_ubyte_d16 v[vgprG2LB+16+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+3] offen offset:12 // load one buffer value
/* g2l=16, load component 13 */
buffer_load_ubyte_d16 v24, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+3] offen offset:13 // load one buffer value
/* g2l=16, load component 14 */
buffer_load_ubyte_d16_hi v25, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+3] offen offset:14 // load one buffer value
/* g2l=16, load component 15 */
buffer_load_ubyte_d16_hi v26, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+3] offen offset:15 // load one buffer value
s_waitcnt vmcnt(14)
v_lshlrev_b32 v12, 0x8, v12                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+16+0], v[vgprG2LB+16+0], v12   // pack a sub 8-bit with dest
s_waitcnt vmcnt(13)
v_or_b32 v[vgprG2LB+16+0], v[vgprG2LB+16+0], v13   // pack a sub 8-bit with dest
s_waitcnt vmcnt(12)
v_lshlrev_b32 v14, 0x8, v14                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+16+0], v[vgprG2LB+16+0], v14   // pack a sub 8-bit with dest
s_waitcnt vmcnt(10)
v_lshlrev_b32 v16, 0x8, v16                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+16+1], v[vgprG2LB+16+1], v16   // pack a sub 8-bit with dest
s_waitcnt vmcnt(9)
v_or_b32 v[vgprG2LB+16+1], v[vgprG2LB+16+1], v17   // pack a sub 8-bit with dest
s_waitcnt vmcnt(8)
v_lshlrev_b32 v18, 0x8, v18                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+16+1], v[vgprG2LB+16+1], v18   // pack a sub 8-bit with dest
s_waitcnt vmcnt(6)
v_lshlrev_b32 v20, 0x8, v20                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+16+2], v[vgprG2LB+16+2], v20   // pack a sub 8-bit with dest
s_waitcnt vmcnt(5)
v_or_b32 v[vgprG2LB+16+2], v[vgprG2LB+16+2], v21   // pack a sub 8-bit with dest
s_waitcnt vmcnt(4)
v_lshlrev_b32 v22, 0x8, v22                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+16+2], v[vgprG2LB+16+2], v22   // pack a sub 8-bit with dest
s_waitcnt vmcnt(2)
v_lshlrev_b32 v24, 0x8, v24                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+16+3], v[vgprG2LB+16+3], v24   // pack a sub 8-bit with dest
s_waitcnt vmcnt(1)
v_or_b32 v[vgprG2LB+16+3], v[vgprG2LB+16+3], v25   // pack a sub 8-bit with dest
s_waitcnt vmcnt(0)
v_lshlrev_b32 v26, 0x8, v26                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+16+3], v[vgprG2LB+16+3], v26   // pack a sub 8-bit with dest
/* g2l=20, load component 0 */
buffer_load_ubyte_d16 v[vgprG2LB+20+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+4] offen offset:0 // load one buffer value
/* g2l=20, load component 1 */
buffer_load_ubyte_d16 v12, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+4] offen offset:1 // load one buffer value
/* g2l=20, load component 2 */
buffer_load_ubyte_d16_hi v13, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+4] offen offset:2 // load one buffer value
/* g2l=20, load component 3 */
buffer_load_ubyte_d16_hi v14, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+4] offen offset:3 // load one buffer value
/* g2l=20, load component 4 */
buffer_load_ubyte_d16 v[vgprG2LB+20+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+4] offen offset:4 // load one buffer value
/* g2l=20, load component 5 */
buffer_load_ubyte_d16 v16, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+4] offen offset:5 // load one buffer value
/* g2l=20, load component 6 */
buffer_load_ubyte_d16_hi v17, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+4] offen offset:6 // load one buffer value
/* g2l=20, load component 7 */
buffer_load_ubyte_d16_hi v18, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+4] offen offset:7 // load one buffer value
/* g2l=20, load component 8 */
buffer_load_ubyte_d16 v[vgprG2LB+20+2], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+4] offen offset:8 // load one buffer value
/* g2l=20, load component 9 */
buffer_load_ubyte_d16 v20, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+4] offen offset:9 // load one buffer value
/* g2l=20, load component 10 */
buffer_load_ubyte_d16_hi v21, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+4] offen offset:10 // load one buffer value
/* g2l=20, load component 11 */
buffer_load_ubyte_d16_hi v22, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+4] offen offset:11 // load one buffer value
/* g2l=20, load component 12 */
buffer_load_ubyte_d16 v[vgprG2LB+20+3], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+4] offen offset:12 // load one buffer value
/* g2l=20, load component 13 */
buffer_load_ubyte_d16 v24, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+4] offen offset:13 // load one buffer value
/* g2l=20, load component 14 */
buffer_load_ubyte_d16_hi v25, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+4] offen offset:14 // load one buffer value
/* g2l=20, load component 15 */
buffer_load_ubyte_d16_hi v26, v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+4] offen offset:15 // load one buffer value
s_waitcnt vmcnt(14)
v_lshlrev_b32 v12, 0x8, v12                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+20+0], v[vgprG2LB+20+0], v12   // pack a sub 8-bit with dest
s_waitcnt vmcnt(13)
v_or_b32 v[vgprG2LB+20+0], v[vgprG2LB+20+0], v13   // pack a sub 8-bit with dest
s_waitcnt vmcnt(12)
v_lshlrev_b32 v14, 0x8, v14                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+20+0], v[vgprG2LB+20+0], v14   // pack a sub 8-bit with dest
s_waitcnt vmcnt(10)
v_lshlrev_b32 v16, 0x8, v16                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+20+1], v[vgprG2LB+20+1], v16   // pack a sub 8-bit with dest
s_waitcnt vmcnt(9)
v_or_b32 v[vgprG2LB+20+1], v[vgprG2LB+20+1], v17   // pack a sub 8-bit with dest
s_waitcnt vmcnt(8)
v_lshlrev_b32 v18, 0x8, v18                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+20+1], v[vgprG2LB+20+1], v18   // pack a sub 8-bit with dest
s_waitcnt vmcnt(6)
v_lshlrev_b32 v20, 0x8, v20                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+20+2], v[vgprG2LB+20+2], v20   // pack a sub 8-bit with dest
s_waitcnt vmcnt(5)
v_or_b32 v[vgprG2LB+20+2], v[vgprG2LB+20+2], v21   // pack a sub 8-bit with dest
s_waitcnt vmcnt(4)
v_lshlrev_b32 v22, 0x8, v22                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+20+2], v[vgprG2LB+20+2], v22   // pack a sub 8-bit with dest
s_waitcnt vmcnt(2)
v_lshlrev_b32 v24, 0x8, v24                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+20+3], v[vgprG2LB+20+3], v24   // pack a sub 8-bit with dest
s_waitcnt vmcnt(1)
v_or_b32 v[vgprG2LB+20+3], v[vgprG2LB+20+3], v25   // pack a sub 8-bit with dest
s_waitcnt vmcnt(0)
v_lshlrev_b32 v26, 0x8, v26                        // shift left to higher 8 bits
v_or_b32 v[vgprG2LB+20+3], v[vgprG2LB+20+3], v26   // pack a sub 8-bit with dest
s_waitcnt vmcnt(0)                                 // 2wait for global read
// Skip force waitcnt0
s_barrier

/* local write a */
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+0:vgprG2LA+0+3] offset:0 // lwoA_0_0_0_0 = (0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0
ds_write_b128 v[vgprLocalWriteAddrA], v[vgprG2LA+4:vgprG2LA+4+3] offset:4608 // lwoA_0_0_1_0 = (0*LSCA)*(MT0I+PAD) + (1*LSPA) = 4608

/* local write b */
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+0:vgprG2LB+0+3] offset:0 // lwoB_0_0_0_0 = (0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+4:vgprG2LB+4+3] offset:4608 // lwoB_0_0_1_0 = (0*LSCB)*(MT1J+PAD) + (1*LSPB) = 4608
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+8:vgprG2LB+8+3] offset:9216 // lwoB_0_0_2_0 = (0*LSCB)*(MT1J+PAD) + (2*LSPB) = 9216
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+12:vgprG2LB+12+3] offset:13824 // lwoB_0_0_3_0 = (0*LSCB)*(MT1J+PAD) + (3*LSPB) = 13824
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+16:vgprG2LB+16+3] offset:18432 // lwoB_0_0_4_0 = (0*LSCB)*(MT1J+PAD) + (4*LSPB) = 18432
ds_write_b128 v[vgprLocalWriteAddrB], v[vgprG2LB+20:vgprG2LB+20+3] offset:23040 // lwoB_0_0_5_0 = (0*LSCB)*(MT1J+PAD) + (5*LSPB) = 23040

/* Recalc local read offsets */
/* lr0I */
v_and_b32 v13, 63, v[vgprSerial]                   // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v12, 15, v13                             // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v12, 0x8, v12                        // 1. N offset: nOffset = nIdx * nStride(256)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v13, 4, v13                          // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshlrev_b32 v13, 0x3, v13                        // 5. K offset: lrKOffset = kIdx * mStride(8)
v_add_u32 v12, v13, v12                            // 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v16, 6, v[vgprSerial]                // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(64)
v_and_b32 v16, 1, v16                              // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshlrev_b32 v16, 0xc, v16                        // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(4096)
v_add_u32 v12, v16, v12                            // 7. final local read offset: flrOffset = lrOffset + WOffset
/* lr1J */
v_and_b32 v14, 63, v[vgprSerial]                   // 0. thread id in wave: wtid = tid % wavelength(64)
v_and_b32 v13, 15, v14                             // 1. N offset: nIdx = wtid % MI_N(16)
v_lshlrev_b32 v13, 0x8, v13                        // 1. N offset: nOffset = nIdx * nStride(256)
/* Skip. 2. block offset: bnOffset = 0 when num1DBlocks = 1 */
                                                   // 4. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)
v_lshrrev_b32 v14, 4, v14                          // 5. K offset: kIdx = wtid / (MIN(16) * MIBB(1))
v_lshlrev_b32 v14, 0x3, v14                        // 5. K offset: lrKOffset = kIdx * mStride(8)
v_add_u32 v13, v14, v13                            // 6. offset in wave: lrOffset = bnOffset + lrKOffset
v_lshrrev_b32 v15, 7, v[vgprSerial]                // 7. wave offset in N dimen: wtid = tid / dividedForWaveId(128)
v_and_b32 v15, 1, v15                              // 7. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)
v_lshlrev_b32 v15, 0xc, v15                        // 7. wave offset in M dimen: wOffset = wtid0 * W0Stride(4096)
v_add_u32 v13, v15, v13                            // 7. final local read offset: flrOffset = lrOffset + WOffset
v_lshrrev_b32 v14, 6, v[vgprSerial]                // v14 = v[vgprSerial] / 64
v_lshrrev_b32 v14, 2, v14                          // LSU offset: Get LSU wave_id
s_mov_b32 s8, 256                                  // LSU offset: stride = lsuStride(256) when umlds==True
v_mul_lo_u32 v14, s8, v14                          // LSU offset: lsuoffset = wave_id*lsuStride*(MT0+PAD)
v_add_u32 v[vgprLocalReadAddrA], v14, v12          // Final Offset: offset = (lro0+lsuoffset)*bpeDS(1)
v_lshrrev_b32 v15, 8, v[vgprLocalReadAddrA]        // Final Offset: padding 32 per block 256
v_lshlrev_b32 v15, 0x5, v15                        // Final Offset: padding 32 per block 256
v_add_u32 v[vgprLocalReadAddrA], v15, v[vgprLocalReadAddrA] // Final Offset: add padding 32 per block 256
/* N/A */
v_lshrrev_b32 v12, 6, v[vgprSerial]                // v12 = v[vgprSerial] / 64
v_lshrrev_b32 v12, 2, v12                          // LSU offset: Get LSU wave_id
                                                   // LSU offset: stride = lsuStride(256) when umlds==True (dup assign opt.)
v_mul_lo_u32 v12, s8, v12                          // LSU offset: lsuoffset = wave_id*lsuStride*(MT1+PAD)
v_add_u32 v[vgprLocalReadAddrB], v12, v13          // Final Offset: offset = (lro1+lsuoffset)*bpeDS(1)
v_lshrrev_b32 v14, 8, v[vgprLocalReadAddrB]        // Final Offset: padding 32 per block 256
v_lshlrev_b32 v14, 0x5, v14                        // Final Offset: padding 32 per block 256
v_add_u32 v[vgprLocalReadAddrB], v14, v[vgprLocalReadAddrB] // Final Offset: add padding 32 per block 256
v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x2400, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)
s_waitcnt lgkmcnt(0)                               // 5wait for local write
// Skip force waitcnt0
s_barrier

/* local read reset offsets a */

/* local read reset offsets b */

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */

/* tail loop: macs */
label_TailLoopBeginL:

/* Tail: remove ValuA/B vgpr buffer [12...76) from pool */

/* Tail: add address/G2L vgpr [76...114) to pool */

/* local read a */
ds_read_b64 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+1], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b64 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+1], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=0 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b64 v[vgprValuB_X0_I0+2:vgprValuB_X0_I0+2+1], v[vgprLocalReadAddrB] offset:9216 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=1 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0
ds_read_b64 v[vgprValuB_X0_I0+4:vgprValuB_X0_I0+4+1], v[vgprLocalReadAddrB] offset:18432 // L -> Reg lro=0 swapByteOffset=0 ti=32 vIdx=2 eIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read inc a */
s_mov_b32 s8, 0x20                                 // inc
v_add_co_u32 v[vgprLocalReadAddrA], vcc, s8, v[vgprLocalReadAddrA] // lrA += 32 (bpeDS)

/* local read inc b */
                                                   // inc (dup assign opt.)
v_add_co_u32 v[vgprLocalReadAddrB], vcc, s8, v[vgprLocalReadAddrB] // lrB += 32 (bpeDS)
s_waitcnt lgkmcnt(0)                               // 4wait for local read
v_and_b32 v76, 63, v[vgprSerial]                   // v76 = v[vgprSerial] % 64
v_lshrrev_b32 v76, 4, v76                          // v76 = v76 / 16
v_lshlrev_b32 v76, 0x3, v76                        // v76 = v76 * 8
v_cmp_ge_i32 s[28:29], v76, s[sgprLoopCounterL]    // check K index >= Size L
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+0], v[vgprValuA_X0_I0+0+0+0+0], 0x0, s[28:29] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0+1], 0x0, s[28:29] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+0], v[vgprValuB_X0_I0+0+0+0+0], 0x0, s[28:29] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+2+0+0+0], v[vgprValuB_X0_I0+2+0+0+0], 0x0, s[28:29] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+0], v[vgprValuB_X0_I0+4+0+0+0], 0x0, s[28:29] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0+1], 0x0, s[28:29] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+2+0+0+1], 0x0, s[28:29] // set 0 if K_idx >= sizeL
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0+1], 0x0, s[28:29] // set 0 if K_idx >= sizeL
v_sub_u32 v76, s[sgprLoopCounterL], v76            // get distance between size and k index
v_cmp_lt_i32 s[28:29], v76, 8                      // set partial 0 if distance less than input per thread
s_and_b32 s30, s[sgprLoopCounterL], 7              // get inputs for edge thread
s_sub_u32 s30, 8, s30                              // use shift to fill 0 for outside element
s_lshl_b32 s30, s30, 3                             // use shift to fill 0 for outside element
v_lshlrev_b64 v[78:79], s30, v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1]
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+0], v[vgprValuA_X0_I0+0+0+0+0], v78, s[28:29]
v_cndmask_b32 v[vgprValuA_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0+1], v79, s[28:29]
v_lshlrev_b64 v[78:79], s30, v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1]
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+0], v[vgprValuB_X0_I0+0+0+0+0], v78, s[28:29]
v_cndmask_b32 v[vgprValuB_X0_I0+0+0+0+1], v[vgprValuB_X0_I0+0+0+0+1], v79, s[28:29]
v_lshlrev_b64 v[78:79], s30, v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1]
v_cndmask_b32 v[vgprValuB_X0_I0+2+0+0+0], v[vgprValuB_X0_I0+2+0+0+0], v78, s[28:29]
v_cndmask_b32 v[vgprValuB_X0_I0+2+0+0+1], v[vgprValuB_X0_I0+2+0+0+1], v79, s[28:29]
v_lshlrev_b64 v[78:79], s30, v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1]
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+0], v[vgprValuB_X0_I0+4+0+0+0], v78, s[28:29]
v_cndmask_b32 v[vgprValuB_X0_I0+4+0+0+1], v[vgprValuB_X0_I0+4+0+0+1], v79, s[28:29]
s_nop 1
v_mfma_i32_16x16x32_i8 v[0:3], v[vgprValuB_X0_I0+0+0+0:vgprValuB_X0_I0+0+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[0:3] // left value = v[0+0:3+0]
v_mfma_i32_16x16x32_i8 v[4:7], v[vgprValuB_X0_I0+2+0+0:vgprValuB_X0_I0+2+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[4:7] // left value = v[4+0:7+0]
v_mfma_i32_16x16x32_i8 v[8:11], v[vgprValuB_X0_I0+4+0+0:vgprValuB_X0_I0+4+0+0+1], v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+1], v[8:11] // left value = v[8+0:11+0]

/* closeLoop loopL finalLoop=1 tailLoop=1 */
s_sub_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0x20 // dec counterL (tailLoop)
s_add_u32 s[sgprOrigLoopCounter], s[sgprOrigLoopCounter], 0x20 // inc counterL
s_cmp_le_i32 s[sgprLoopCounterL], 0x0              // counterL<=0
s_cbranch_scc0 label_TailLoopBeginL                // restart LoopL
label_TailLoopEndL:
label_SkipTailLoopL:

/* Tail: remove address/G2L [76...114) from pool */
label_Summation_End_PX36ZW0DRIGRU763_0:
/* endSummation: add vgpr [12...114) to pool */
.set sgprWGM, UNDEF
.set sgprLoopCounterL, UNDEF
.set sgprOrigLoopCounter, UNDEF
.set sgprAddressA, UNDEF
.set sgprAddressB, UNDEF
.set sgprStridesA, UNDEF
.set sgprStridesB, UNDEF
.set sgprStaggerUIter, UNDEF
.set sgprSrdA, UNDEF
.set sgprSrdB, UNDEF
.set sgprShadowLimitA, UNDEF
.set sgprShadowLimitB, UNDEF
.set sgprWrapUA, UNDEF
.set sgprWrapUB, UNDEF
.set sgprGlobalReadIncsA, UNDEF
.set sgprGlobalReadIncsB, UNDEF
.set sgprScalarGlobalReadOffsetA, UNDEF
.set sgprScalarGlobalReadOffsetB, UNDEF
/* load store sgprs */
.set sgprAddressScaleAlphaVec, 28
.set sgpractivationAlpha, 30
.set sgpractivationBeta, 31
.set sgprActivationType, 32
s_and_b32 s8, s[sgprGSU], 0x3fff                   // Restore GSU
s_cmp_eq_u32 s8, 1                                 // GSU == 1 ?
s_cbranch_scc0 label_GSU_4                         // branch if GSU != 1
/* Check if custom structure pointer is null */
s_cmp_eq_u32 s[sgprArgType], 2                     // ArgType == 2 ?
s_cbranch_scc1 label_LoadExternalEpilogueStruct_1  // branch if ArgType == 2
s_load_dwordx4 s[28:31], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58
s_load_dword s32, s[sgprKernArgAddress:sgprKernArgAddress+1], 0x68
s_branch label_LoadExternalEpilogueStructEnd_1
label_LoadExternalEpilogueStruct_1:
s_load_dwordx2 s[28:29], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x90
s_load_dwordx2 s[30:31], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xb8
s_load_dword s32, s[sgprKernArgAddress:sgprKernArgAddress+1], 0xc0
label_LoadExternalEpilogueStructEnd_1:
label_GSU_4:
.set sgprSrdScaleAlphaVec, 40

/* Mapping of Acc register -> C Vgpr register */

/* Multiply MI out register with Alpha -> C Vgpr register */

/* not-LocalSplitU: global write indices */
/* computeStoreVgprs */
v_lshrrev_b32 v16, 6, v[vgprSerial]                // v16 = v[vgprSerial] / 64
v_lshrrev_b32 v17, 1, v16                          // v17 = v16 / 2
v_mul_lo_u32 v17, 0x10, v17                        // wave coordination offset 1
v_and_b32 v13, 63, v[vgprSerial]                   // v13 = v[vgprSerial] % 64
v_lshrrev_b32 v13, 4, v13                          // v13 = v13 / 16
v_lshlrev_b32 v13, 0x2, v13                        // thread0 * continuous_output
v_add_lshl_u32 v13, v17, v13, 0                    // coordination 1 = vwB *(wave_id1 + tid1)
v_mul_lo_u32 v14, v13, s[sgprStrideC1J]            //  offset 1
v_mul_lo_u32 v15, v13, s[sgprStrideD1J]            //  offset 1
v_and_b32 v12, 1, v16                              // v12 = v16 % 2
v_mul_lo_u32 v12, 0x10, v12                        // wave coordination offset 0
v_and_b32 v17, 15, v[vgprSerial]                   // v17 = v[vgprSerial] % 16
v_add_lshl_u32 v12, v17, v12, 0                    // coordination 0 = vwA * (wave_id0 + tid0)
s_mul_i32 s8, 32, s[sgprWorkGroup0]                // wgp0 * MT0
v_add_u32 v12, s8, v12                             // coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0
s_mul_i32 s8, 96, s[sgprWorkGroup1]                // wgp1 * MT1
v_add_u32 v13, s8, v13                             // coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1

/* not-LocalSplitU: global write */

/******************************************/
/* Global Write Elements                  */
/******************************************/
s_waitcnt lgkmcnt(0)                               // wait for 20 bytes of kern args.
s_and_b32 s8, s[sgprGSU], 0x3fff                   // Restore GSU
s_cmp_eq_u32 s8, 1                                 // GSU == 1 ?
s_cbranch_scc1 label_GSU_5                         // branch if GSU == 1
.set sgprAddressScaleAlphaVec, UNDEF
.set sgprSrdScaleAlphaVec, UNDEF
s_and_b32 s40, 31, s[sgprSizeI]                    // s40 = s[sgprSizeI] % 32
s_add_u32 s41, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s41                // wg0 >= nwg0-1 ?
s_cselect_b32 s40, s40, 0                          // set rMT0
s_cmpk_gt_u32 s40, 0x0                             // rMT0 > 0
s_cbranch_scc1 label_GW_B0_E1                      // jump if edges required
s_mov_b32 s43, 0x0                                 // STATIC_DIV: divisior=96
s_mul_i32 s42, 0x555, s[sgprSizeJ]                 // tmp1 = dividend * magic hi
s_lshl_b64 s[42:43], s[42:43], 0x10                // left shift 16 bits
s_mul_i32 s41, s[sgprSizeJ], 0x5556                // tmp0 = dividend * magic lo
s_add_u32 s42, s41, s42                            // add lo
s_addc_u32 s43, s43, 0x0                           // add hi
s_lshr_b64 s[42:43], s[42:43], 0x21                // tmp1 = (dividend * magic) << shift
s_mov_b32 s41, s42                                 // quotient
s_mul_i32 s42, s41, 0x60                           // quotient*divisor
s_sub_u32 s40, s[sgprSizeJ], s42                   // rReg = dividend - quotient*divisor
s_add_u32 s41, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s41                // wg1 >= nwg1-1
s_cselect_b32 s40, s40, 0                          // set rMT1
s_cmpk_gt_u32 s40, 0x0                             // rMT1 > 0
s_cbranch_scc1 label_GW_B0_E1                      // jump if edges required
label_GW_B0_E0_1:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,1,0:vw1); (0,0,2,0:vw1); (0,0,3,0:vw1); (1,0,0,0:vw1); (1,0,1,0:vw1); (1,0,2,0:vw1); (1,0,3,0:vw1); (2,0,0,0:vw1); (2,0,1,0:vw1); (2,0,2,0:vw1); (2,0,3,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,1,0,0) */
/* (d1,vc1,d0,vc0)=(1,2,0,0) */
/* (d1,vc1,d0,vc0)=(1,3,0,0) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
/* (d1,vc1,d0,vc0)=(2,1,0,0) */
/* (d1,vc1,d0,vc0)=(2,2,0,0) */
/* (d1,vc1,d0,vc0)=(2,3,0,0) */
v_add_lshl_u32 v22, v15, v12, 0x2                  // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=12, coord0Vgpr=12

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0), (1, 0, 0, 0), (1, 0, 1, 0), (1, 0, 2, 0), (1, 0, 3, 0), (2, 0, 0, 0), (2, 0, 1, 0), (2, 0, 2, 0), (2, 0, 3, 0)] */
v_mov_b32 v[vgprValuC+24], v[vgprValuC+0]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+25], v[vgprValuC+1]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+26], v[vgprValuC+2]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+27], v[vgprValuC+3]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+28], v[vgprValuC+4]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+29], v[vgprValuC+5]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+30], v[vgprValuC+6]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+31], v[vgprValuC+7]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+32], v[vgprValuC+8]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+33], v[vgprValuC+9]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+34], v[vgprValuC+10]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+35], v[vgprValuC+11]         // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_dword v24, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v25, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v26, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v27, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s12, s[sgprStrideD1J], 116               // scale StrideD *= numRows(29) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v28, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v29, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v30, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v31, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_mul_i32 s12, s[sgprStrideD1J], 116               // scale StrideD *= numRows(29) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v32, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v33, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v34, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_lshl_b32 s12, s[sgprStrideD1J], 2                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s12        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_dword v35, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_B0_E1:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,1,0:vw1); (0,0,2,0:vw1); (0,0,3,0:vw1); (1,0,0,0:vw1); (1,0,1,0:vw1); (1,0,2,0:vw1); (1,0,3,0:vw1); (2,0,0,0:vw1); (2,0,1,0:vw1); (2,0,2,0:vw1); (2,0,3,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v46, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v22, v15, v12, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v22, v46, v22, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v13, vcc, v13, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v14, v14, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v15, v15, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v24, v15, v12, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v24, v46, v24, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v13, vcc, v13, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v14, v14, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v15, v15, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v26, v15, v12, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v26, v46, v26, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v13, vcc, v13, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v14, v14, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v15, v15, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v28, v15, v12, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v28, v46, v28, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v13, vcc, v13, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s48, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v14, v14, s48                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s48, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v15, v15, s48                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v30, v15, v12, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v30, v46, v30, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,1,0,0) */
v_add_co_u32 v13, vcc, v13, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v14, v14, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v15, v15, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v32, v15, v12, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v32, v46, v32, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,2,0,0) */
v_add_co_u32 v13, vcc, v13, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v14, v14, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v15, v15, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v34, v15, v12, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v34, v46, v34, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,3,0,0) */
v_add_co_u32 v13, vcc, v13, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v14, v14, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v15, v15, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v36, v15, v12, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v36, v46, v36, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v13, vcc, v13, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s48, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v14, v14, s48                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s48, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v15, v15, s48                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v38, v15, v12, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v38, v46, v38, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,1,0,0) */
v_add_co_u32 v13, vcc, v13, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v14, v14, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v15, v15, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v40, v15, v12, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v40, v46, v40, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,2,0,0) */
v_add_co_u32 v13, vcc, v13, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v14, v14, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v15, v15, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v42, v15, v12, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v42, v46, v42, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,3,0,0) */
v_add_co_u32 v13, vcc, v13, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v14, v14, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v15, v15, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v44, v15, v12, 0x2                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v44, v46, v44, s[52:53]              // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0), (1, 0, 0, 0), (1, 0, 1, 0), (1, 0, 2, 0), (1, 0, 3, 0), (2, 0, 0, 0), (2, 0, 1, 0), (2, 0, 2, 0), (2, 0, 3, 0)] */
v_mov_b32 v[vgprValuC+23], v[vgprValuC+0]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+25], v[vgprValuC+1]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+27], v[vgprValuC+2]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+29], v[vgprValuC+3]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+31], v[vgprValuC+4]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+33], v[vgprValuC+5]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+35], v[vgprValuC+6]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+37], v[vgprValuC+7]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+39], v[vgprValuC+8]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+41], v[vgprValuC+9]          // Rearrange MI out reg
v_mov_b32 v[vgprValuC+43], v[vgprValuC+10]         // Rearrange MI out reg
v_mov_b32 v[vgprValuC+45], v[vgprValuC+11]         // Rearrange MI out reg

/* apply mask, calc new C and issue writes */
buffer_store_dword v23, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v25, v24, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v27, v26, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v29, v28, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v31, v30, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v33, v32, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v35, v34, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v37, v36, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v39, v38, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v41, v40, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v43, v42, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
buffer_store_dword v45, v44, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_1                            // jump to end
label_GW_End_1:
s_getpc_b64 s[40:41]                               // addr of next instr
s_add_i32 s42, label_KernelEnd, 0x4                // target branch offset
s_add_u32 s40, s40, s42                            // add target branch offset
s_addc_u32 s41, s41, 0                             // add high and carry
s_setpc_b64 s[40:41]                               // branch to label_KernelEnd
label_GSU_5:
.set sgprAddressScaleAlphaVec, 28
.set sgprSrdScaleAlphaVec, 40
s_mov_b32 s[sgprSrdScaleAlphaVec+0], s[sgprAddressScaleAlphaVec+0] // init SRD base address (lower)
s_mov_b32 s[sgprSrdScaleAlphaVec+1], s[sgprAddressScaleAlphaVec+1] // init SRD base address (upper) + other fields
s_mov_b32 s[sgprSrdScaleAlphaVec+3], Srd127_96     // Set bits 127_96 in post-loop SRD
s_cmp_eq_u64 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], 0 // s[AddressScaleAlphaVec] == 0 ?
s_cbranch_scc0 label_ScaleAlphaVec_1AddrValid      // branch if s[AddressScaleAlphaVec] != 0
s_mov_b32 s[sgprSrdScaleAlphaVec+2], 0
s_branch label_ScaleAlphaVec_1AddrValid_End
label_ScaleAlphaVec_1AddrValid:
s_mov_b32 s[sgprSrdScaleAlphaVec+2], s[sgprSizeI]
label_ScaleAlphaVec_1AddrValid_End:

s_mul_i32 s[sgprSrdScaleAlphaVec+2], 0x4, s[sgprSrdScaleAlphaVec+2] // ScaleAlphaVec scaled by BPE

/******************************************/
/* Read vector to LDS                     */
/******************************************/
s_mul_i32 s11, 32, s[sgprWorkGroup0]               // wgp0 * MT0
v_add_u32 v18, s11, v[vgprSerial]                  // coord 0 = wgp0 * MT0 + thread offset
v_lshlrev_b32 v17, 0x2, v18                        // Global scaleAlpha address scaled by BPE
s_mul_i32 s11, 96, s[sgprWorkGroup1]               // wgp1 * MT1
v_add_u32 v18, s11, v[vgprSerial]                  // coord 1 = wgp1 * MT1 + thread offset
buffer_load_dword v16, v17, s[sgprSrdScaleAlphaVec:sgprSrdScaleAlphaVec+3], 0 offen offset:0 // Load ScaleAlphaVec
v_lshlrev_b32 v18, 0x2, v[vgprSerial]              // Local address scaled by BPE
s_barrier                                          // wait for all global loads.
v_cmp_gt_u32 s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1], s[sgprSrdScaleAlphaVec+2], 0 //  == 0 ?
s_waitcnt vmcnt(0)                                 // wait for global load
v_cndmask_b32 v16, 1, v16, s[sgprAddressScaleAlphaVec:sgprAddressScaleAlphaVec+1] // 1. mul 1 if 0
ds_write_b32 v18, v16 offset:0                     // store scaleAlpha
.set sgprAddressScaleAlphaVec, UNDEF
.set sgprSrdScaleAlphaVec, UNDEF
s_cmpk_eq_u32 s[sgprActivationType], 5             // activationType == 5
s_cbranch_scc1 label_To_Activation_Relu_VW1_1      // Branch if true
label_To_Activation_None_VW1_1:
s_getpc_b64 s[12:13]                               // addr of next instr
s_add_i32 s8, label_Activation_None_VW1, 0x4       // target branch offset
s_add_u32 s12, s12, s8                             // add target branch offset
s_addc_u32 s13, s13, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd_1
label_To_Activation_Relu_VW1_1:
s_getpc_b64 s[12:13]                               // addr of next instr
s_add_i32 s8, label_Activation_Relu_VW1, 0x4       // target branch offset
s_add_u32 s12, s12, s8                             // add target branch offset
s_addc_u32 s13, s13, 0                             // add high and carry
s_branch label_ActivationSetPCAddrEnd_1
label_ActivationSetPCAddrEnd_1:
s_cmpk_eq_u32 s[sgprBeta], 0x0                     // Beta == 0
s_cbranch_scc0 label_GW_Beta_2                     // Branch if Beta is not zero

s_and_b32 s40, 31, s[sgprSizeI]                    // s40 = s[sgprSizeI] % 32
s_add_u32 s41, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s41                // wg0 >= nwg0-1 ?
s_cselect_b32 s40, s40, 0                          // set rMT0
s_cmpk_gt_u32 s40, 0x0                             // rMT0 > 0
s_cbranch_scc1 label_GW_B0_E1_1                    // jump if edges required
s_mov_b32 s43, 0x0                                 // STATIC_DIV: divisior=96
s_mul_i32 s42, 0x555, s[sgprSizeJ]                 // tmp1 = dividend * magic hi
s_lshl_b64 s[42:43], s[42:43], 0x10                // left shift 16 bits
s_mul_i32 s41, s[sgprSizeJ], 0x5556                // tmp0 = dividend * magic lo
s_add_u32 s42, s41, s42                            // add lo
s_addc_u32 s43, s43, 0x0                           // add hi
s_lshr_b64 s[42:43], s[42:43], 0x21                // tmp1 = (dividend * magic) << shift
s_mov_b32 s41, s42                                 // quotient
s_mul_i32 s42, s41, 0x60                           // quotient*divisor
s_sub_u32 s40, s[sgprSizeJ], s42                   // rReg = dividend - quotient*divisor
s_add_u32 s41, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s41                // wg1 >= nwg1-1
s_cselect_b32 s40, s40, 0                          // set rMT1
s_cmpk_gt_u32 s40, 0x0                             // rMT1 > 0
s_cbranch_scc1 label_GW_B0_E1_1                    // jump if edges required
label_GW_B0_E0_2:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,1,0:vw1); (0,0,2,0:vw1); (0,0,3,0:vw1); (1,0,0,0:vw1); (1,0,1,0:vw1); (1,0,2,0:vw1); (1,0,3,0:vw1); (2,0,0,0:vw1); (2,0,1,0:vw1); (2,0,2,0:vw1); (2,0,3,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
s_mul_i32 s34, 32, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v24, v12, s34
v_lshlrev_b32 v24, 0x2, v24                        // ScaleAlpha address scaled by BPE
s_waitcnt lgkmcnt(0)                               // Wait for LDS write
s_barrier                                          // LDS write barrier
ds_read_b32 v25, v24 offset:0                      // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
/* (d1,vc1,d0,vc0)=(1,1,0,0) */
/* (d1,vc1,d0,vc0)=(1,2,0,0) */
/* (d1,vc1,d0,vc0)=(1,3,0,0) */
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
/* (d1,vc1,d0,vc0)=(2,1,0,0) */
/* (d1,vc1,d0,vc0)=(2,2,0,0) */
/* (d1,vc1,d0,vc0)=(2,3,0,0) */
v_add_lshl_u32 v22, v15, v12, 0x0                  // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=12, coord0Vgpr=12

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0), (1, 0, 0, 0), (1, 0, 1, 0), (1, 0, 2, 0), (1, 0, 3, 0), (2, 0, 0, 0), (2, 0, 1, 0), (2, 0, 2, 0), (2, 0, 3, 0)] */
v_mul_lo_u32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+29], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+30], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+31], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+32], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+33], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+34], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+35], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+36], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+37], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */

s_waitcnt lgkmcnt(0)                               // lgkmcnt(0) = 1 - 1 (scaleAlphaVec) (interleaved)
v_mul_lo_u32 v[vgprValuC+26], v25, v[vgprValuC+26] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v26
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v26, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+26], v[vgprValuC+26], s34, v18 // x= min(127, max(-128, x))
buffer_store_byte v26, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+27], v25, v[vgprValuC+27] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v27
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v27, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+27], v[vgprValuC+27], s34, v18 // x= min(127, max(-128, x))
s_lshl_b32 s34, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v27, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+28], v25, v[vgprValuC+28] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v28
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v28, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+28], v[vgprValuC+28], s34, v18 // x= min(127, max(-128, x))
s_lshl_b32 s34, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v28, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+29], v25, v[vgprValuC+29] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v29
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v29, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+29], v[vgprValuC+29], s34, v18 // x= min(127, max(-128, x))
s_lshl_b32 s34, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v29, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+30], v25, v[vgprValuC+30] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v30
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v30, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+30], v[vgprValuC+30], s34, v18 // x= min(127, max(-128, x))
s_mul_i32 s34, s[sgprStrideD1J], 29                // scale StrideD *= numRows(29) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v30, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+31], v25, v[vgprValuC+31] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v31
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v31, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+31], v[vgprValuC+31], s34, v18 // x= min(127, max(-128, x))
s_lshl_b32 s34, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v31, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+32], v25, v[vgprValuC+32] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v32
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v32, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+32], v[vgprValuC+32], s34, v18 // x= min(127, max(-128, x))
s_lshl_b32 s34, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v32, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+33], v25, v[vgprValuC+33] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v33
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v33, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+33], v[vgprValuC+33], s34, v18 // x= min(127, max(-128, x))
s_lshl_b32 s34, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v33, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+34], v25, v[vgprValuC+34] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v34
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v34, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+34], v[vgprValuC+34], s34, v18 // x= min(127, max(-128, x))
s_mul_i32 s34, s[sgprStrideD1J], 29                // scale StrideD *= numRows(29) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v34, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+35], v25, v[vgprValuC+35] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v35
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v35, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+35], v[vgprValuC+35], s34, v18 // x= min(127, max(-128, x))
s_lshl_b32 s34, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v35, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+36], v25, v[vgprValuC+36] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v36
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v36, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+36], v[vgprValuC+36], s34, v18 // x= min(127, max(-128, x))
s_lshl_b32 s34, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v36, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+37], v25, v[vgprValuC+37] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v37
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v37, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+37], v[vgprValuC+37], s34, v18 // x= min(127, max(-128, x))
s_lshl_b32 s34, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v37, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_B0_E1_1:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,1,0:vw1); (0,0,2,0:vw1); (0,0,3,0:vw1); (1,0,0,0:vw1); (1,0,1,0:vw1); (1,0,2,0:vw1); (1,0,3,0:vw1); (2,0,0,0:vw1); (2,0,1,0:vw1); (2,0,2,0:vw1); (2,0,3,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v59, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
s_mul_i32 s48, 32, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v23, v12, s48
v_lshlrev_b32 v23, 0x2, v23                        // ScaleAlpha address scaled by BPE
s_waitcnt lgkmcnt(0)                               // Wait for LDS write
s_barrier                                          // LDS write barrier
ds_read_b32 v24, v23 offset:0                      // load scaleAlpha
v_add_lshl_u32 v22, v15, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v22, v59, v22, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v13, vcc, v13, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v14, v14, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v15, v15, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
s_mul_i32 s48, 32, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v27, v12, s48
v_lshlrev_b32 v27, 0x2, v27                        // ScaleAlpha address scaled by BPE
v_add_lshl_u32 v26, v15, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v26, v59, v26, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v13, vcc, v13, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v14, v14, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v15, v15, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
s_mul_i32 s48, 32, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v30, v12, s48
v_lshlrev_b32 v30, 0x2, v30                        // ScaleAlpha address scaled by BPE
v_add_lshl_u32 v29, v15, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v29, v59, v29, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v13, vcc, v13, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v14, v14, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v15, v15, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
s_mul_i32 s48, 32, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v33, v12, s48
v_lshlrev_b32 v33, 0x2, v33                        // ScaleAlpha address scaled by BPE
v_add_lshl_u32 v32, v15, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v32, v59, v32, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v13, vcc, v13, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s48, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v14, v14, s48                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s48, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v15, v15, s48                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
s_mul_i32 s48, 32, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v36, v12, s48
v_lshlrev_b32 v36, 0x2, v36                        // ScaleAlpha address scaled by BPE
v_add_lshl_u32 v35, v15, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v35, v59, v35, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,1,0,0) */
v_add_co_u32 v13, vcc, v13, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v14, v14, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v15, v15, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
s_mul_i32 s48, 32, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v39, v12, s48
v_lshlrev_b32 v39, 0x2, v39                        // ScaleAlpha address scaled by BPE
v_add_lshl_u32 v38, v15, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v38, v59, v38, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,2,0,0) */
v_add_co_u32 v13, vcc, v13, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v14, v14, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v15, v15, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
s_mul_i32 s48, 32, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v42, v12, s48
v_lshlrev_b32 v42, 0x2, v42                        // ScaleAlpha address scaled by BPE
v_add_lshl_u32 v41, v15, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v41, v59, v41, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,3,0,0) */
v_add_co_u32 v13, vcc, v13, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v14, v14, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v15, v15, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
s_mul_i32 s48, 32, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v45, v12, s48
v_lshlrev_b32 v45, 0x2, v45                        // ScaleAlpha address scaled by BPE
v_add_lshl_u32 v44, v15, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v44, v59, v44, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v13, vcc, v13, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s48, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v14, v14, s48                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s48, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v15, v15, s48                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
s_mul_i32 s48, 32, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v48, v12, s48
v_lshlrev_b32 v48, 0x2, v48                        // ScaleAlpha address scaled by BPE
v_add_lshl_u32 v47, v15, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v47, v59, v47, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,1,0,0) */
v_add_co_u32 v13, vcc, v13, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v14, v14, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v15, v15, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
s_mul_i32 s48, 32, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v51, v12, s48
v_lshlrev_b32 v51, 0x2, v51                        // ScaleAlpha address scaled by BPE
v_add_lshl_u32 v50, v15, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v50, v59, v50, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,2,0,0) */
v_add_co_u32 v13, vcc, v13, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v14, v14, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v15, v15, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
s_mul_i32 s48, 32, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v54, v12, s48
v_lshlrev_b32 v54, 0x2, v54                        // ScaleAlpha address scaled by BPE
v_add_lshl_u32 v53, v15, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v53, v59, v53, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,3,0,0) */
v_add_co_u32 v13, vcc, v13, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v14, v14, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v15, v15, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
s_mul_i32 s48, 32, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v57, v12, s48
v_lshlrev_b32 v57, 0x2, v57                        // ScaleAlpha address scaled by BPE
v_add_lshl_u32 v56, v15, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v56, v59, v56, s[52:53]              // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0), (1, 0, 0, 0), (1, 0, 1, 0), (1, 0, 2, 0), (1, 0, 3, 0), (2, 0, 0, 0), (2, 0, 1, 0), (2, 0, 2, 0), (2, 0, 3, 0)] */
v_mul_lo_u32 v[vgprValuC+25], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+31], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+34], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+37], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+40], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+43], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+46], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+49], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+52], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+55], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+58], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
s_waitcnt lgkmcnt(0)                               // wait for ScaleAlphaVec

/* apply mask, calc new C and issue writes */
v_mul_lo_u32 v[vgprValuC+25], v24, v[vgprValuC+25] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v25
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v25, v16
s_movk_i32 s48, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+25], v[vgprValuC+25], s48, v18 // x= min(127, max(-128, x))
buffer_store_byte v25, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+28], v24, v[vgprValuC+28] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v28
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v28, v16
s_movk_i32 s48, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+28], v[vgprValuC+28], s48, v18 // x= min(127, max(-128, x))
buffer_store_byte v28, v26, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+31], v24, v[vgprValuC+31] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v31
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v31, v16
s_movk_i32 s48, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+31], v[vgprValuC+31], s48, v18 // x= min(127, max(-128, x))
buffer_store_byte v31, v29, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+34], v24, v[vgprValuC+34] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v34
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v34, v16
s_movk_i32 s48, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+34], v[vgprValuC+34], s48, v18 // x= min(127, max(-128, x))
buffer_store_byte v34, v32, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+37], v24, v[vgprValuC+37] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v37
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v37, v16
s_movk_i32 s48, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+37], v[vgprValuC+37], s48, v18 // x= min(127, max(-128, x))
buffer_store_byte v37, v35, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+40], v24, v[vgprValuC+40] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v40
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v40, v16
s_movk_i32 s48, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+40], v[vgprValuC+40], s48, v18 // x= min(127, max(-128, x))
buffer_store_byte v40, v38, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+43], v24, v[vgprValuC+43] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v43
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v43, v16
s_movk_i32 s48, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+43], v[vgprValuC+43], s48, v18 // x= min(127, max(-128, x))
buffer_store_byte v43, v41, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+46], v24, v[vgprValuC+46] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v46
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v46, v16
s_movk_i32 s48, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+46], v[vgprValuC+46], s48, v18 // x= min(127, max(-128, x))
buffer_store_byte v46, v44, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+49], v24, v[vgprValuC+49] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v49
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v49, v16
s_movk_i32 s48, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+49], v[vgprValuC+49], s48, v18 // x= min(127, max(-128, x))
buffer_store_byte v49, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+52], v24, v[vgprValuC+52] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v52
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v52, v16
s_movk_i32 s48, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+52], v[vgprValuC+52], s48, v18 // x= min(127, max(-128, x))
buffer_store_byte v52, v50, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+55], v24, v[vgprValuC+55] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v55
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v55, v16
s_movk_i32 s48, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+55], v[vgprValuC+55], s48, v18 // x= min(127, max(-128, x))
buffer_store_byte v55, v53, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+58], v24, v[vgprValuC+58] // *= ScaleAlphaVecVMul
v_mov_b32 v16, v58
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v58, v16
s_movk_i32 s48, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+58], v[vgprValuC+58], s48, v18 // x= min(127, max(-128, x))
buffer_store_byte v58, v56, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_Beta_2:
s_and_b32 s40, 31, s[sgprSizeI]                    // s40 = s[sgprSizeI] % 32
s_add_u32 s41, -0x1, s[sgprNumWorkGroups0]
s_cmp_ge_u32 s[sgprWorkGroup0], s41                // wg0 >= nwg0-1 ?
s_cselect_b32 s40, s40, 0                          // set rMT0
s_cmpk_gt_u32 s40, 0x0                             // rMT0 > 0
s_cbranch_scc1 label_GW_B1_E1                      // jump if edges required
s_mov_b32 s43, 0x0                                 // STATIC_DIV: divisior=96
s_mul_i32 s42, 0x555, s[sgprSizeJ]                 // tmp1 = dividend * magic hi
s_lshl_b64 s[42:43], s[42:43], 0x10                // left shift 16 bits
s_mul_i32 s41, s[sgprSizeJ], 0x5556                // tmp0 = dividend * magic lo
s_add_u32 s42, s41, s42                            // add lo
s_addc_u32 s43, s43, 0x0                           // add hi
s_lshr_b64 s[42:43], s[42:43], 0x21                // tmp1 = (dividend * magic) << shift
s_mov_b32 s41, s42                                 // quotient
s_mul_i32 s42, s41, 0x60                           // quotient*divisor
s_sub_u32 s40, s[sgprSizeJ], s42                   // rReg = dividend - quotient*divisor
s_add_u32 s41, -0x1, s[sgprNumWorkGroups1]
s_cmp_ge_u32 s[sgprWorkGroup1], s41                // wg1 >= nwg1-1
s_cselect_b32 s40, s40, 0                          // set rMT1
s_cmpk_gt_u32 s40, 0x0                             // rMT1 > 0
s_cbranch_scc1 label_GW_B1_E1                      // jump if edges required
label_GW_B1_E0:

/* edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 factorDim=0 */

/******************************************/
/* Global Write Beta Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,1,0:vw1); (0,0,2,0:vw1); (0,0,3,0:vw1); (1,0,0,0:vw1); (1,0,1,0:vw1); (1,0,2,0:vw1); (1,0,3,0:vw1); (2,0,0,0:vw1); (2,0,1,0:vw1); (2,0,2,0:vw1); (2,0,3,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_add_lshl_u32 v23, v14, v12, 0x0                  // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=12, coord0Vgpr=12
buffer_load_ubyte_d16 v25, v23, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s34, 32, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v24, v12, s34
v_lshlrev_b32 v24, 0x2, v24                        // ScaleAlpha address scaled by BPE
s_waitcnt lgkmcnt(0)                               // Wait for LDS write
s_barrier                                          // LDS write barrier
ds_read_b32 v26, v24 offset:0                      // load scaleAlpha
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
s_lshl_b32 s34, s[sgprStrideC1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_ubyte_d16 v28, v23, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
s_lshl_b32 s34, s[sgprStrideC1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_ubyte_d16 v30, v23, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
s_lshl_b32 s34, s[sgprStrideC1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_ubyte_d16 v32, v23, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
s_mul_i32 s34, s[sgprStrideC1J], 29                // scale StrideC *= numRows(29) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_ubyte_d16 v34, v23, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(1,1,0,0) */
s_lshl_b32 s34, s[sgprStrideC1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_ubyte_d16 v36, v23, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(1,2,0,0) */
s_lshl_b32 s34, s[sgprStrideC1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_ubyte_d16 v38, v23, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(1,3,0,0) */
s_lshl_b32 s34, s[sgprStrideC1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_ubyte_d16 v40, v23, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
s_mul_i32 s34, s[sgprStrideC1J], 29                // scale StrideC *= numRows(29) * bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_ubyte_d16 v42, v23, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(2,1,0,0) */
s_lshl_b32 s34, s[sgprStrideC1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_ubyte_d16 v44, v23, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(2,2,0,0) */
s_lshl_b32 s34, s[sgprStrideC1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_ubyte_d16 v46, v23, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
/* (d1,vc1,d0,vc0)=(2,3,0,0) */
s_lshl_b32 s34, s[sgprStrideC1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_load_ubyte_d16 v48, v23, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
v_add_lshl_u32 v22, v15, v12, 0x0                  // optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=12, coord0Vgpr=12

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0), (1, 0, 0, 0), (1, 0, 1, 0), (1, 0, 2, 0), (1, 0, 3, 0), (2, 0, 0, 0), (2, 0, 1, 0), (2, 0, 2, 0), (2, 0, 3, 0)] */
v_mul_lo_u32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+29], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+31], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+33], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+35], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+37], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+39], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+41], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+43], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+45], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+47], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+49], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha

/* apply mask, calc new C and issue writes */

s_waitcnt lgkmcnt(0), vmcnt(11)                    // vmcnt(11) = 12 - 1 (beta) lgkmcnt(0) = 1 - 1 (scaleAlphaVec) (interleaved)
v_mul_lo_u32 v[vgprValuC+27], v26, v[vgprValuC+27] // *= ScaleAlphaVecVMul
v_mov_b32 v17, 0x0                                 // value = 0
v_bfe_i32 v16, v25, v17, 8                         // int8 to int32
v_mul_lo_u32 v16, s[sgprBeta], v16                 // C = C*beta
v_add_u32 v[vgprValuC+27], v16, v[vgprValuC+27]    // finalSum = sum*alpha + C*beta
v_mov_b32 v16, v27
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v27, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+27], v[vgprValuC+27], s34, v18 // x= min(127, max(-128, x))
buffer_store_byte v27, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(11)                                // vmcnt(10) = 12 - 2 (beta) (interleaved)
v_mul_lo_u32 v[vgprValuC+29], v26, v[vgprValuC+29] // *= ScaleAlphaVecVMul
v_mov_b32 v17, 0x0                                 // value = 0
v_bfe_i32 v16, v28, v17, 8                         // int8 to int32
v_mul_lo_u32 v16, s[sgprBeta], v16                 // C = C*beta
v_add_u32 v[vgprValuC+29], v16, v[vgprValuC+29]    // finalSum = sum*alpha + C*beta
v_mov_b32 v16, v29
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v29, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+29], v[vgprValuC+29], s34, v18 // x= min(127, max(-128, x))
s_lshl_b32 s34, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v29, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(11)                                // vmcnt(9) = 12 - 3 (beta) (interleaved)
v_mul_lo_u32 v[vgprValuC+31], v26, v[vgprValuC+31] // *= ScaleAlphaVecVMul
v_mov_b32 v17, 0x0                                 // value = 0
v_bfe_i32 v16, v30, v17, 8                         // int8 to int32
v_mul_lo_u32 v16, s[sgprBeta], v16                 // C = C*beta
v_add_u32 v[vgprValuC+31], v16, v[vgprValuC+31]    // finalSum = sum*alpha + C*beta
v_mov_b32 v16, v31
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v31, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+31], v[vgprValuC+31], s34, v18 // x= min(127, max(-128, x))
s_lshl_b32 s34, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v31, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(11)                                // vmcnt(8) = 12 - 4 (beta) (interleaved)
v_mul_lo_u32 v[vgprValuC+33], v26, v[vgprValuC+33] // *= ScaleAlphaVecVMul
v_mov_b32 v17, 0x0                                 // value = 0
v_bfe_i32 v16, v32, v17, 8                         // int8 to int32
v_mul_lo_u32 v16, s[sgprBeta], v16                 // C = C*beta
v_add_u32 v[vgprValuC+33], v16, v[vgprValuC+33]    // finalSum = sum*alpha + C*beta
v_mov_b32 v16, v33
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v33, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+33], v[vgprValuC+33], s34, v18 // x= min(127, max(-128, x))
s_lshl_b32 s34, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v33, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(11)                                // vmcnt(7) = 12 - 5 (beta) (interleaved)
v_mul_lo_u32 v[vgprValuC+35], v26, v[vgprValuC+35] // *= ScaleAlphaVecVMul
v_mov_b32 v17, 0x0                                 // value = 0
v_bfe_i32 v16, v34, v17, 8                         // int8 to int32
v_mul_lo_u32 v16, s[sgprBeta], v16                 // C = C*beta
v_add_u32 v[vgprValuC+35], v16, v[vgprValuC+35]    // finalSum = sum*alpha + C*beta
v_mov_b32 v16, v35
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v35, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+35], v[vgprValuC+35], s34, v18 // x= min(127, max(-128, x))
s_mul_i32 s34, s[sgprStrideD1J], 29                // scale StrideD *= numRows(29) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v35, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(11)                                // vmcnt(6) = 12 - 6 (beta) (interleaved)
v_mul_lo_u32 v[vgprValuC+37], v26, v[vgprValuC+37] // *= ScaleAlphaVecVMul
v_mov_b32 v17, 0x0                                 // value = 0
v_bfe_i32 v16, v36, v17, 8                         // int8 to int32
v_mul_lo_u32 v16, s[sgprBeta], v16                 // C = C*beta
v_add_u32 v[vgprValuC+37], v16, v[vgprValuC+37]    // finalSum = sum*alpha + C*beta
v_mov_b32 v16, v37
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v37, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+37], v[vgprValuC+37], s34, v18 // x= min(127, max(-128, x))
s_lshl_b32 s34, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v37, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(11)                                // vmcnt(5) = 12 - 7 (beta) (interleaved)
v_mul_lo_u32 v[vgprValuC+39], v26, v[vgprValuC+39] // *= ScaleAlphaVecVMul
v_mov_b32 v17, 0x0                                 // value = 0
v_bfe_i32 v16, v38, v17, 8                         // int8 to int32
v_mul_lo_u32 v16, s[sgprBeta], v16                 // C = C*beta
v_add_u32 v[vgprValuC+39], v16, v[vgprValuC+39]    // finalSum = sum*alpha + C*beta
v_mov_b32 v16, v39
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v39, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+39], v[vgprValuC+39], s34, v18 // x= min(127, max(-128, x))
s_lshl_b32 s34, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v39, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(11)                                // vmcnt(4) = 12 - 8 (beta) (interleaved)
v_mul_lo_u32 v[vgprValuC+41], v26, v[vgprValuC+41] // *= ScaleAlphaVecVMul
v_mov_b32 v17, 0x0                                 // value = 0
v_bfe_i32 v16, v40, v17, 8                         // int8 to int32
v_mul_lo_u32 v16, s[sgprBeta], v16                 // C = C*beta
v_add_u32 v[vgprValuC+41], v16, v[vgprValuC+41]    // finalSum = sum*alpha + C*beta
v_mov_b32 v16, v41
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v41, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+41], v[vgprValuC+41], s34, v18 // x= min(127, max(-128, x))
s_lshl_b32 s34, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v41, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(11)                                // vmcnt(3) = 12 - 9 (beta) (interleaved)
v_mul_lo_u32 v[vgprValuC+43], v26, v[vgprValuC+43] // *= ScaleAlphaVecVMul
v_mov_b32 v17, 0x0                                 // value = 0
v_bfe_i32 v16, v42, v17, 8                         // int8 to int32
v_mul_lo_u32 v16, s[sgprBeta], v16                 // C = C*beta
v_add_u32 v[vgprValuC+43], v16, v[vgprValuC+43]    // finalSum = sum*alpha + C*beta
v_mov_b32 v16, v43
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v43, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+43], v[vgprValuC+43], s34, v18 // x= min(127, max(-128, x))
s_mul_i32 s34, s[sgprStrideD1J], 29                // scale StrideD *= numRows(29) * bpe
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v43, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(11)                                // vmcnt(2) = 12 - 10 (beta) (interleaved)
v_mul_lo_u32 v[vgprValuC+45], v26, v[vgprValuC+45] // *= ScaleAlphaVecVMul
v_mov_b32 v17, 0x0                                 // value = 0
v_bfe_i32 v16, v44, v17, 8                         // int8 to int32
v_mul_lo_u32 v16, s[sgprBeta], v16                 // C = C*beta
v_add_u32 v[vgprValuC+45], v16, v[vgprValuC+45]    // finalSum = sum*alpha + C*beta
v_mov_b32 v16, v45
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v45, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+45], v[vgprValuC+45], s34, v18 // x= min(127, max(-128, x))
s_lshl_b32 s34, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v45, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(11)                                // vmcnt(1) = 12 - 11 (beta) (interleaved)
v_mul_lo_u32 v[vgprValuC+47], v26, v[vgprValuC+47] // *= ScaleAlphaVecVMul
v_mov_b32 v17, 0x0                                 // value = 0
v_bfe_i32 v16, v46, v17, 8                         // int8 to int32
v_mul_lo_u32 v16, s[sgprBeta], v16                 // C = C*beta
v_add_u32 v[vgprValuC+47], v16, v[vgprValuC+47]    // finalSum = sum*alpha + C*beta
v_mov_b32 v16, v47
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v47, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+47], v[vgprValuC+47], s34, v18 // x= min(127, max(-128, x))
s_lshl_b32 s34, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v47, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D

s_waitcnt vmcnt(11)                                // vmcnt(0) = 12 - 12 (beta) (interleaved)
v_mul_lo_u32 v[vgprValuC+49], v26, v[vgprValuC+49] // *= ScaleAlphaVecVMul
v_mov_b32 v17, 0x0                                 // value = 0
v_bfe_i32 v16, v48, v17, 8                         // int8 to int32
v_mul_lo_u32 v16, s[sgprBeta], v16                 // C = C*beta
v_add_u32 v[vgprValuC+49], v16, v[vgprValuC+49]    // finalSum = sum*alpha + C*beta
v_mov_b32 v16, v49
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v49, v16
s_movk_i32 s34, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+49], v[vgprValuC+49], s34, v18 // x= min(127, max(-128, x))
s_lshl_b32 s34, s[sgprStrideD1J], 0                // incToNextRow: Scale by BPE
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s34        // incToNextRow: gra SRD += inc(lower)
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], 0         // incToNextRow: gra SRD += inc(upper)
buffer_store_byte v49, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_B1_E1:

/* edge=1, allocate 6 sgpr. perBatchTmpS=4 perBatchMaskS=2 perElementMaskS=0 elementsPerBatch=16 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Edge_Mask optSrdIncForRow=0 factorDim=0 */

/******************************************/
/* Global Write Beta Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1); (0,0,1,0:vw1); (0,0,2,0:vw1); (0,0,3,0:vw1); (1,0,0,0:vw1); (1,0,1,0:vw1); (1,0,2,0:vw1); (1,0,3,0:vw1); (2,0,0,0:vw1); (2,0,1,0:vw1); (2,0,2,0:vw1); (2,0,3,0:vw1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
v_mov_b32 v71, BufferOOB
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v22, v14, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v22, v71, v22, s[52:53]              // LDC clip if OOB. offset
buffer_load_ubyte_d16 v24, v22, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s48, 32, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v23, v12, s48
v_lshlrev_b32 v23, 0x2, v23                        // ScaleAlpha address scaled by BPE
s_waitcnt lgkmcnt(0)                               // Wait for LDS write
s_barrier                                          // LDS write barrier
ds_read_b32 v25, v23 offset:0                      // load scaleAlpha
v_add_lshl_u32 v22, v15, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v22, v71, v22, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,1,0,0) */
v_add_co_u32 v13, vcc, v13, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v14, v14, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v15, v15, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v27, v14, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v27, v71, v27, s[52:53]              // LDC clip if OOB. offset
buffer_load_ubyte_d16 v29, v27, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s48, 32, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v28, v12, s48
v_lshlrev_b32 v28, 0x2, v28                        // ScaleAlpha address scaled by BPE
v_add_lshl_u32 v27, v15, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v27, v71, v27, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,2,0,0) */
v_add_co_u32 v13, vcc, v13, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v14, v14, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v15, v15, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v31, v14, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v31, v71, v31, s[52:53]              // LDC clip if OOB. offset
buffer_load_ubyte_d16 v33, v31, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s48, 32, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v32, v12, s48
v_lshlrev_b32 v32, 0x2, v32                        // ScaleAlpha address scaled by BPE
v_add_lshl_u32 v31, v15, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v31, v71, v31, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(0,3,0,0) */
v_add_co_u32 v13, vcc, v13, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v14, v14, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v15, v15, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v35, v14, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v35, v71, v35, s[52:53]              // LDC clip if OOB. offset
buffer_load_ubyte_d16 v37, v35, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s48, 32, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v36, v12, s48
v_lshlrev_b32 v36, 0x2, v36                        // ScaleAlpha address scaled by BPE
v_add_lshl_u32 v35, v15, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v35, v71, v35, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,0,0,0) */
v_add_co_u32 v13, vcc, v13, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s48, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v14, v14, s48                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s48, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v15, v15, s48                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v39, v14, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v39, v71, v39, s[52:53]              // LDC clip if OOB. offset
buffer_load_ubyte_d16 v41, v39, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s48, 32, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v40, v12, s48
v_lshlrev_b32 v40, 0x2, v40                        // ScaleAlpha address scaled by BPE
v_add_lshl_u32 v39, v15, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v39, v71, v39, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,1,0,0) */
v_add_co_u32 v13, vcc, v13, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v14, v14, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v15, v15, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v43, v14, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v43, v71, v43, s[52:53]              // LDC clip if OOB. offset
buffer_load_ubyte_d16 v45, v43, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s48, 32, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v44, v12, s48
v_lshlrev_b32 v44, 0x2, v44                        // ScaleAlpha address scaled by BPE
v_add_lshl_u32 v43, v15, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v43, v71, v43, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,2,0,0) */
v_add_co_u32 v13, vcc, v13, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v14, v14, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v15, v15, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v47, v14, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v47, v71, v47, s[52:53]              // LDC clip if OOB. offset
buffer_load_ubyte_d16 v49, v47, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s48, 32, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v48, v12, s48
v_lshlrev_b32 v48, 0x2, v48                        // ScaleAlpha address scaled by BPE
v_add_lshl_u32 v47, v15, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v47, v71, v47, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(1,3,0,0) */
v_add_co_u32 v13, vcc, v13, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v14, v14, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v15, v15, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v51, v14, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v51, v71, v51, s[52:53]              // LDC clip if OOB. offset
buffer_load_ubyte_d16 v53, v51, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s48, 32, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v52, v12, s48
v_lshlrev_b32 v52, 0x2, v52                        // ScaleAlpha address scaled by BPE
v_add_lshl_u32 v51, v15, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v51, v71, v51, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,0,0,0) */
v_add_co_u32 v13, vcc, v13, 29                     // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
s_mul_i32 s48, s[sgprStrideC1J], 29                // scale stride
v_add_i32 v14, v14, s48                            // ROWINC- Move cinRowPtr to next row
s_mul_i32 s48, s[sgprStrideD1J], 29                // scale stride
v_add_i32 v15, v15, s48                            // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v55, v14, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v55, v71, v55, s[52:53]              // LDC clip if OOB. offset
buffer_load_ubyte_d16 v57, v55, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s48, 32, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v56, v12, s48
v_lshlrev_b32 v56, 0x2, v56                        // ScaleAlpha address scaled by BPE
v_add_lshl_u32 v55, v15, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v55, v71, v55, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,1,0,0) */
v_add_co_u32 v13, vcc, v13, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v14, v14, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v15, v15, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v59, v14, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v59, v71, v59, s[52:53]              // LDC clip if OOB. offset
buffer_load_ubyte_d16 v61, v59, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s48, 32, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v60, v12, s48
v_lshlrev_b32 v60, 0x2, v60                        // ScaleAlpha address scaled by BPE
v_add_lshl_u32 v59, v15, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v59, v71, v59, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,2,0,0) */
v_add_co_u32 v13, vcc, v13, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v14, v14, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v15, v15, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v63, v14, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v63, v71, v63, s[52:53]              // LDC clip if OOB. offset
buffer_load_ubyte_d16 v65, v63, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s48, 32, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v64, v12, s48
v_lshlrev_b32 v64, 0x2, v64                        // ScaleAlpha address scaled by BPE
v_add_lshl_u32 v63, v15, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v63, v71, v63, s[52:53]              // LDD clip if OOB. offset
/* (d1,vc1,d0,vc0)=(2,3,0,0) */
v_add_co_u32 v13, vcc, v13, 1                      // coord1.1: coord1Vgpr += d1*sg1*VW + vc1

/* Fix for UseInitialStridesCD, emitAddressSetupCode */
v_add_u32 v14, v14, s[sgprStrideC1J]               // ROWINC- Move cinRowPtr to next row
v_add_u32 v15, v15, s[sgprStrideD1J]               // Move coutRowPtrD to next row
v_cmp_lt_u32 s[48:49], v12, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[52:53], v13, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[52:53], s[48:49], s[52:53]             // in0 && in1
v_add_lshl_u32 v67, v14, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v67, v71, v67, s[52:53]              // LDC clip if OOB. offset
buffer_load_ubyte_d16 v69, v67, s[sgprSrdC:sgprSrdC+3], 0 offen offset:0 // load C
s_mul_i32 s48, 32, s[sgprWorkGroup0]               // wgp0 * MT0
v_sub_u32 v68, v12, s48
v_lshlrev_b32 v68, 0x2, v68                        // ScaleAlpha address scaled by BPE
v_add_lshl_u32 v67, v15, v12, 0x0                  // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v67, v71, v67, s[52:53]              // LDD clip if OOB. offset

/* rC *= alpha batchElements=[(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0), (1, 0, 0, 0), (1, 0, 1, 0), (1, 0, 2, 0), (1, 0, 3, 0), (2, 0, 0, 0), (2, 0, 1, 0), (2, 0, 2, 0), (2, 0, 3, 0)] */
v_mul_lo_u32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+0] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+30], s[sgprAlpha], v[vgprValuC+1] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+34], s[sgprAlpha], v[vgprValuC+2] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+38], s[sgprAlpha], v[vgprValuC+3] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+42], s[sgprAlpha], v[vgprValuC+4] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+46], s[sgprAlpha], v[vgprValuC+5] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+50], s[sgprAlpha], v[vgprValuC+6] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+54], s[sgprAlpha], v[vgprValuC+7] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+58], s[sgprAlpha], v[vgprValuC+8] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+62], s[sgprAlpha], v[vgprValuC+9] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+66], s[sgprAlpha], v[vgprValuC+10] // Multiply MI out reg with alpha
v_mul_lo_u32 v[vgprValuC+70], s[sgprAlpha], v[vgprValuC+11] // Multiply MI out reg with alpha
s_waitcnt 0                                        // wait for Beta, ScaleAlphaVec

/* apply mask, calc new C and issue writes */
v_mul_lo_u32 v[vgprValuC+26], v25, v[vgprValuC+26] // *= ScaleAlphaVecVMul
v_mov_b32 v17, 0x0                                 // value = 0
v_bfe_i32 v16, v24, v17, 8                         // int8 to int32
v_mul_lo_u32 v16, s[sgprBeta], v16                 // C = C*beta
v_add_u32 v[vgprValuC+26], v16, v[vgprValuC+26]    // finalSum = sum*alpha + C*beta
v_mov_b32 v16, v26
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v26, v16
s_movk_i32 s48, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+26], v[vgprValuC+26], s48, v18 // x= min(127, max(-128, x))
buffer_store_byte v26, v22, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+30], v25, v[vgprValuC+30] // *= ScaleAlphaVecVMul
v_mov_b32 v17, 0x0                                 // value = 0
v_bfe_i32 v16, v29, v17, 8                         // int8 to int32
v_mul_lo_u32 v16, s[sgprBeta], v16                 // C = C*beta
v_add_u32 v[vgprValuC+30], v16, v[vgprValuC+30]    // finalSum = sum*alpha + C*beta
v_mov_b32 v16, v30
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v30, v16
s_movk_i32 s48, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+30], v[vgprValuC+30], s48, v18 // x= min(127, max(-128, x))
buffer_store_byte v30, v27, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+34], v25, v[vgprValuC+34] // *= ScaleAlphaVecVMul
v_mov_b32 v17, 0x0                                 // value = 0
v_bfe_i32 v16, v33, v17, 8                         // int8 to int32
v_mul_lo_u32 v16, s[sgprBeta], v16                 // C = C*beta
v_add_u32 v[vgprValuC+34], v16, v[vgprValuC+34]    // finalSum = sum*alpha + C*beta
v_mov_b32 v16, v34
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v34, v16
s_movk_i32 s48, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+34], v[vgprValuC+34], s48, v18 // x= min(127, max(-128, x))
buffer_store_byte v34, v31, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+38], v25, v[vgprValuC+38] // *= ScaleAlphaVecVMul
v_mov_b32 v17, 0x0                                 // value = 0
v_bfe_i32 v16, v37, v17, 8                         // int8 to int32
v_mul_lo_u32 v16, s[sgprBeta], v16                 // C = C*beta
v_add_u32 v[vgprValuC+38], v16, v[vgprValuC+38]    // finalSum = sum*alpha + C*beta
v_mov_b32 v16, v38
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v38, v16
s_movk_i32 s48, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+38], v[vgprValuC+38], s48, v18 // x= min(127, max(-128, x))
buffer_store_byte v38, v35, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+42], v25, v[vgprValuC+42] // *= ScaleAlphaVecVMul
v_mov_b32 v17, 0x0                                 // value = 0
v_bfe_i32 v16, v41, v17, 8                         // int8 to int32
v_mul_lo_u32 v16, s[sgprBeta], v16                 // C = C*beta
v_add_u32 v[vgprValuC+42], v16, v[vgprValuC+42]    // finalSum = sum*alpha + C*beta
v_mov_b32 v16, v42
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v42, v16
s_movk_i32 s48, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+42], v[vgprValuC+42], s48, v18 // x= min(127, max(-128, x))
buffer_store_byte v42, v39, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+46], v25, v[vgprValuC+46] // *= ScaleAlphaVecVMul
v_mov_b32 v17, 0x0                                 // value = 0
v_bfe_i32 v16, v45, v17, 8                         // int8 to int32
v_mul_lo_u32 v16, s[sgprBeta], v16                 // C = C*beta
v_add_u32 v[vgprValuC+46], v16, v[vgprValuC+46]    // finalSum = sum*alpha + C*beta
v_mov_b32 v16, v46
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v46, v16
s_movk_i32 s48, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+46], v[vgprValuC+46], s48, v18 // x= min(127, max(-128, x))
buffer_store_byte v46, v43, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+50], v25, v[vgprValuC+50] // *= ScaleAlphaVecVMul
v_mov_b32 v17, 0x0                                 // value = 0
v_bfe_i32 v16, v49, v17, 8                         // int8 to int32
v_mul_lo_u32 v16, s[sgprBeta], v16                 // C = C*beta
v_add_u32 v[vgprValuC+50], v16, v[vgprValuC+50]    // finalSum = sum*alpha + C*beta
v_mov_b32 v16, v50
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v50, v16
s_movk_i32 s48, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+50], v[vgprValuC+50], s48, v18 // x= min(127, max(-128, x))
buffer_store_byte v50, v47, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+54], v25, v[vgprValuC+54] // *= ScaleAlphaVecVMul
v_mov_b32 v17, 0x0                                 // value = 0
v_bfe_i32 v16, v53, v17, 8                         // int8 to int32
v_mul_lo_u32 v16, s[sgprBeta], v16                 // C = C*beta
v_add_u32 v[vgprValuC+54], v16, v[vgprValuC+54]    // finalSum = sum*alpha + C*beta
v_mov_b32 v16, v54
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v54, v16
s_movk_i32 s48, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+54], v[vgprValuC+54], s48, v18 // x= min(127, max(-128, x))
buffer_store_byte v54, v51, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+58], v25, v[vgprValuC+58] // *= ScaleAlphaVecVMul
v_mov_b32 v17, 0x0                                 // value = 0
v_bfe_i32 v16, v57, v17, 8                         // int8 to int32
v_mul_lo_u32 v16, s[sgprBeta], v16                 // C = C*beta
v_add_u32 v[vgprValuC+58], v16, v[vgprValuC+58]    // finalSum = sum*alpha + C*beta
v_mov_b32 v16, v58
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v58, v16
s_movk_i32 s48, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+58], v[vgprValuC+58], s48, v18 // x= min(127, max(-128, x))
buffer_store_byte v58, v55, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+62], v25, v[vgprValuC+62] // *= ScaleAlphaVecVMul
v_mov_b32 v17, 0x0                                 // value = 0
v_bfe_i32 v16, v61, v17, 8                         // int8 to int32
v_mul_lo_u32 v16, s[sgprBeta], v16                 // C = C*beta
v_add_u32 v[vgprValuC+62], v16, v[vgprValuC+62]    // finalSum = sum*alpha + C*beta
v_mov_b32 v16, v62
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v62, v16
s_movk_i32 s48, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+62], v[vgprValuC+62], s48, v18 // x= min(127, max(-128, x))
buffer_store_byte v62, v59, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+66], v25, v[vgprValuC+66] // *= ScaleAlphaVecVMul
v_mov_b32 v17, 0x0                                 // value = 0
v_bfe_i32 v16, v65, v17, 8                         // int8 to int32
v_mul_lo_u32 v16, s[sgprBeta], v16                 // C = C*beta
v_add_u32 v[vgprValuC+66], v16, v[vgprValuC+66]    // finalSum = sum*alpha + C*beta
v_mov_b32 v16, v66
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v66, v16
s_movk_i32 s48, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+66], v[vgprValuC+66], s48, v18 // x= min(127, max(-128, x))
buffer_store_byte v66, v63, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
v_mul_lo_u32 v[vgprValuC+70], v25, v[vgprValuC+70] // *= ScaleAlphaVecVMul
v_mov_b32 v17, 0x0                                 // value = 0
v_bfe_i32 v16, v69, v17, 8                         // int8 to int32
v_mul_lo_u32 v16, s[sgprBeta], v16                 // C = C*beta
v_add_u32 v[vgprValuC+70], v16, v[vgprValuC+70]    // finalSum = sum*alpha + C*beta
v_mov_b32 v16, v70
s_swappc_b64 s[28:29], s[12:13]
v_mov_b32 v70, v16
s_movk_i32 s48, -0x80                              // -128
v_mov_b32 v18, 0x7f                                // 127
v_med3_i32 v[vgprValuC+70], v[vgprValuC+70], s48, v18 // x= min(127, max(-128, x))
buffer_store_byte v70, v67, s[sgprSrdD:sgprSrdD+3], 0 offen offset:0 // store D
s_nop 0                                            // 1 wait state required when next inst writes vgprs held by previous dwordx4 store inst
s_branch label_GW_End_2                            // jump to end
label_GW_End_2:
label_KernelEnd:
s_endpgm                                           // Kernel End
label_Activation_None_VW1:
s_setpc_b64 s[28:29]
label_Activation_Relu_VW1:
v_max_i32 v16, v16, 0                              // x = max(0, x)
s_setpc_b64 s[28:29]
s_endpgm
label_ASM_End:  /// The end of the kernel
