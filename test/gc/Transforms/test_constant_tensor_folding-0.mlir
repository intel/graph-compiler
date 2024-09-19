// RUN: gc-opt --split-input-file -pass-pipeline="builtin.module(constant-tensor-folding)" %s | FileCheck %s

// COM：A complete example of compile-time and runtime folding.

// CHECK-LABEL: func.func @entry
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6 floordiv 2, d5, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
module {
  // COM: A three-layer mlp. %arg0: input feature. %arg1, %arg2, %arg3: weight of #1, #2 and #3 linear.
  func.func @entry(%arg0: tensor<64x32xbf16>, %arg2: tensor<32x256xbf16>, %arg3: tensor<256x1024xbf16>) 
      -> tensor<64x1024xbf16> attributes {llvm.emit_c_interface, compiletime_const_args_index = [1 : i32], runtime_const_args_index = [2 : i32]} {
    %1 = tensor.empty() : tensor<2x1x32x32xbf16>
    %packed_arg0 = tensor.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %1 : tensor<64x32xbf16> -> tensor<2x1x32x32xbf16>

    %arg1 = arith.constant dense<"0x99571CBE05BA1C3D926AFCBD782B34BE67A737BEBF181ABE3C4E253B5D32F73D1566963D9B6F313EB74C0FBD430B253AE8E2E23DB0CBC53C46C014BE2E0981BD4D313F3E833D37BEAB70E13D65B6CA3DB194983D1E60983D950B71BD1815FDBB32DF9A3DD106EBBDB4A8233E841EC3BDE8C7C13D3734073EFF067DBD070206BEF6AF633DB209843C2135C3BD4F85B83C43BD1CBE04841A3E3E78BD3DE9D0D0BCF660093ED074083E14D43E3ECDA735BE8C8C0E3E40C60FBE4F73C9BDB4358DBD263D323C64E61EBEE535D23D238F013C727EA73DBDBAA1BD79D53EBE392981BDC06B453D10E37D3D2D2B41BEE1FA6BBD410E513D05588BBD514AB0BB0624243E3D80993C8E6A113EE57CFD3D23FE37BE001573BD86AD143E7F052D3E97C07DBD19B4113D3E87F6BDB971E83DFEA12BBC5D51F9BD4F203A3ED454043E22775BBD2EE8313EB027D03D8FEFD7BD0E56B7BDBF963FBE5B64E93D9291FBBD027101BE573DFD3D0CD6EB3D809B863DA9E8263E9EF2A43D717AB73D3CF597BD9FB7243DC603003D61780E3E3992293D8B1B25BE6B0024BE806DCB3D5BAB91BD9A33AFBDD5BC3BBE6D920FBE0D90F53D4513383E2219A0BBE8B6FBBD341C42BD42F235BED91A1ABDC3AEB0BD5AC1383DE0EADC3D303D11BE850D263E8281163E5CB78A3D19EB34BE33150F3E84F8EE3D18FC823DB26CCBBD09AB06BED909FFBA605EFE3B9014B7BD1606DA3D75ACE13D0910753C33C6843DE9951CBECD220ABD0EF2BF3D14BB2E3C798718BD60A53A3E8B83E53D18663DBE4D07CABD37CE043EA6B18E3D3D0F303EE392073EC92A1ABED6900E3E72D3E73D8CEF803D1B4D3D3E997D283E210F923BC2D131BECEAF913DB981EFBDCBCCCCBA2B6711BE4E32FE3C5D5D33BD2F34313EB7EC48BC26CDFD3D07170B3E1CD816BE310DD2BD9E03023E1EA8F3BD8B99EEBBFC97433E047F8DBDDD6BA03DA3B2433E34D7C0BC7FDB89BA1980333EF3FC8D3DC05C203E9C7213BD8385403E2F971A3E4357CF3DB39BFBBC784FF8BC7DBD0C3E8301E23D77BF1ABB04F3243CFBA3B1BD5A46C6BD1745A8BDD6950ABD939CC5BDB4226EBCAC622EBD6748FBBDAFF9D53DF29D433E41991C3D4DD7353EE2EF8E3D21EF3B3DF679973D31DEFDBDF0AF303E8D34DFBB31B895BD6A633A3EACE125BEE94E95BDA58043BEC9F233BE915F03BD1B7C8F3DE1D367BDD7BBD63D6E990A3E23222F3D4B6CD73DB869C53D8697383E3A86853D973F2C3EFC3827BC4E87FA3DD5903BBE4BB8403E34A9A33D41C8843D4BC8FABD3CD5E8BD4946233D955052BDA5F841BC6C81AFBD5DD8883DB71A753CD0A1263D88690ABE35DAA73CA3557D3D8C09D23D5A27273DECEFDBBCD220023EE036ACBD6CD2443E8F630FBEBC43B73DF03AA4BDC709133E1B94E73D362CE4BCB15F33BE3139443E5FCF62BD0E3C1B3EE99DF93D9E1BB3BA70DB213E38EBDDBC47F10CBEF817293DAD3DEB3B730942BE535C87BD448D7B3B1C8094BD97962B3D5B0F3B3EA3F42A3E4ED46DBD6D72C33C687CC63DEA34C53D1CCC3EBEDCA640BE638ABCBD4B63AFBDA699063E92861E3E98219FBC8E0B233ED3ED573DC856B8BD13880F3EFA0763BD5A8C89BD194519BE89C6CF3D73A219BC5ECBD43D41EFA33D27D8493D756B1ABEC796C93D9A25133C6A5A363E13FB8DBD601755BD3935FABD14D6883D0EF2D33DB8E914BD527347397200433DE72A3F3B62C52F3ED164EF3CD8806FBD05528B3D89701EBE0A09C23DA19B103D05922EBE7A100E3E31C0503D8ED53BBE08463E3E5168013E55F3E53D782EC53DA8BBD93C1711223E05FDB2BDA740113EA27A20BD1685A23D7E35293E02BD8B3CC43F163E4AE6613DE4280F3EEEF20BBE965C1DBEFAAD233E75754E3D96C33BBCB6D7013E0D8E7ABD703C82BDEA0875BC6F57A6BCE83609BE8A8EB53DAB7D3C3E39A50ABEB878A33D9FCEA1BC124AD33C22C34A3DB5F338BE0307BF3C2F0881BD7E15E8BDBEE8C8BDBBFFA63C342F303E15B1CCBB2590153EEA05EF3DE778F2BCE9E1233ECEC244BDBF92D5BDECDEAE3C29750CBDD969FCBD7DC236BE571D1DBEC8FA7DBC243BAD3C38673D3ED15943BEFE4D913D5329273E18AB2EBE19AB5F3D30A62F3E94303CBE1421DABCBE6E133E355D073EEC76633DEB2AB83DA2BF16BC9A46C2BD4EB47EBC4C82343EC1D1E63D13D314BED232E3BD3E5CF1BDC78F9EBD6483233E7290293E514A163E255F0FBE1AEF7BBD5259173EF12524BEDF47793C886BE8BD57B408BE351980BD0FF71ABD24643ABEA79920BED2603A3EEB75393EC6D52B3E458B29BC22C45ABC02BB40BCED4BDEBCA6E9CABC11FB213EC4FB363E5AC2DCBDAD6B4F3CBB85B1BD8093343E487518BEDFA316BD7FFFAEBB9375963DF68A88BD6876013C9FA1C63D95CDB23C911721BE04B5F9BD1B7C8F3DE1D367BDD7BBD63D6E990A3E23222F3D4B6CD73DB869C53D8697383E3A86853D973F2C3EFC3827BC4E87FA3DD5903BBE4BB8403E34A9A33D41C8843D4BC8FABD3CD5E8BD4946233D955052BDA5F841BC6C81AFBD5DD8883DB71A753CD0A1263D88690ABE35DAA73CA3557D3D8C09D23D5A27273DECEFDBBCD220023EE036ACBD6CD2443E8F630FBEBC43B73DF03AA4BDC709133E1B94E73D362CE4BCB15F33BE3139443E5FCF62BD0E3C1B3EE99DF93D9E1BB3BA70DB213E38EBDDBC47F10CBEF817293DAD3997D283E210F923BC2D131BECEAF913DB981EFBDCBCCCCBA2B6711BE4E32FE3C5D5D33BD2F34313EB7EC48BC26CDFD3D07170B3E1CD816BE310DD2BD9E03023E1EA8F3BD8B99EEBBFC97433E047F8DBDDD6BA03DA3B2433E34D7C0BC7FDB89BA1980333EF3EB7EC48B383DE0E383DE0E383DE0E383DE0"> : tensor<32x32xbf16>
    %2 = tensor.empty() : tensor<1x1x32x32xbf16>
    %packed_arg1 = tensor.pack %arg1 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %2 : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    %3 = tensor.empty() : tensor<1x1x16x32x2xbf16>
    %packed_packed_arg1 = tensor.pack %packed_arg1 inner_dims_pos = [2] inner_tiles = [2] into %3 : tensor<1x1x32x32xbf16> -> tensor<1x1x16x32x2xbf16>

    %4 = tensor.empty() : tensor<2x1x32x32xbf16>
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %5 = linalg.fill ins(%cst_0 : bf16) outs(%4 : tensor<2x1x32x32xbf16>) -> tensor<2x1x32x32xbf16>
    %6 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%packed_arg0, %packed_packed_arg1 : tensor<2x1x32x32xbf16>, tensor<1x1x16x32x2xbf16>) outs(%5 : tensor<2x1x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %44 = arith.mulf %in, %in_0 : bf16
      %55 = arith.addf %out, %44 : bf16
      linalg.yield %55 : bf16
    } -> tensor<2x1x32x32xbf16>

    %7 = tensor.empty() : tensor<8x1x32x32xbf16>
    %packed_arg2 = tensor.pack %arg2 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %7 : tensor<32x256xbf16> -> tensor<8x1x32x32xbf16>
    %8 = tensor.empty() : tensor<8x1x16x32x2xbf16>
    %packed_packed_arg2 = tensor.pack %packed_arg2 inner_dims_pos = [2] inner_tiles = [2] into %8 : tensor<8x1x32x32xbf16> -> tensor<8x1x16x32x2xbf16>
    %9 = tensor.empty() : tensor<2x8x32x32xbf16>
    %10 = linalg.fill ins(%cst_0 : bf16) outs(%9 : tensor<2x8x32x32xbf16>) -> tensor<2x8x32x32xbf16>
    %11 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%6, %packed_packed_arg2 : tensor<2x1x32x32xbf16>, tensor<8x1x16x32x2xbf16>) outs(%10 : tensor<2x8x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %44 = arith.mulf %in, %in_0 : bf16
      %55 = arith.addf %out, %44 : bf16
      linalg.yield %55 : bf16
    } -> tensor<2x8x32x32xbf16>

    %12 = tensor.empty() : tensor<32x8x32x32xbf16>
    %packed_arg3 = tensor.pack %arg3 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %12 : tensor<256x1024xbf16> -> tensor<32x8x32x32xbf16>
    %13 = tensor.empty() : tensor<32x8x16x32x2xbf16>
    %packed_packed_arg3 = tensor.pack %packed_arg3 inner_dims_pos = [2] inner_tiles = [2] into %13 : tensor<32x8x32x32xbf16> -> tensor<32x8x16x32x2xbf16>

    %14 = tensor.empty() : tensor<2x32x32x32xbf16>
    %15 = linalg.fill ins(%cst_0 : bf16) outs(%14 : tensor<2x32x32x32xbf16>) -> tensor<2x32x32x32xbf16>
    %16 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%11, %packed_packed_arg3 : tensor<2x8x32x32xbf16>, tensor<32x8x16x32x2xbf16>) outs(%15 : tensor<2x32x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %46 = arith.mulf %in, %in_0 : bf16
      %56 = arith.addf %out, %46 : bf16
      linalg.yield %56 : bf16
    } -> tensor<2x32x32x32xbf16>

    %17 = tensor.empty() : tensor<64x1024xbf16>
    %unpack = tensor.unpack %16 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %17 : tensor<2x32x32x32xbf16> -> tensor<64x1024xbf16>
    return %unpack : tensor<64x1024xbf16>
  }
}

// COM: If enable compile time folding,
// COM: 1 pack in entry for input feature, 
// COM: 4 packs in compiletime_fold for 2 weights, 
// COM: 2 packs in runtime_fold for 1 weights:
// COM: CHECK: tensor.pack
// COM: CHECK: func.func @compiletime_fold
// COM: CHECK: tensor.pack
// COM: CHECK: tensor.pack
// COM: CHECK: tensor.pack
// COM: CHECK: tensor.pack
// COM: CHECK: func.func @runtime_fold
// COM: CHECK: tensor.pack
// COM: CHECK: tensor.pack

// COM: else,
// CHECK: tensor.pack
// CHECK: func.func @runtime_fold
// CHECK: tensor.pack
// CHECK: tensor.pack
// CHECK: tensor.pack
// CHECK: tensor.pack
// CHECK: tensor.pack
// CHECK: tensor.pack
