"?P
BHostIDLE"IDLE1     ̟@A     ̟@a?R?v???i?R?v????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?P@9     ?P@A     ?P@I     ?P@aO:4௛?i????????Unknown?
dHostDataset"Iterator::Model(1      @@9      @@A      5@I      5@a??	Yw???i?p{o????Unknown
oHost_FusedMatMul"sequential/dense/Relu(1      1@9      1@A      1@I      1@aJM٨?|?i??"?|
???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      .@9      .@A      .@I      .@a`????+y?i????<???Unknown
iHostWriteSummary"WriteSummary(1      *@9      *@A      *@I      *@au=? c?u?i???th???Unknown?
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      *@9      *@A      *@I      *@au=? c?u?i???d????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1      &@9      &@A      &@I      &@a??"@ur?i??&??????Unknown
?	HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1      &@9      &@A      &@I      &@a??"@ur?i@.Oe?????Unknown
s
HostDataset"Iterator::Model::ParallelMapV2(1      &@9      &@A      &@I      &@a??"@ur?i?sw?????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1      $@9      $@A      $@I      $@a?????p?iV?Bd$???Unknown
?HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      $@9      $@A      $@I      $@a?????p?iQ8???E???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      "@9      "@A      "@I      "@a??~O:4n?i?>?'d???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1      "@9      "@A      "@I      "@a??~O:4n?i?5?\????Unknown
gHostStridedSlice"strided_slice(1      "@9      "@A      "@I      "@a??~O:4n?i???N?????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1       @9       @A       @I       @aU_c?j?i?@fi????Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @aU_c?j?ip??}B????Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1       @9       @A       @I       @aU_c?j?i??????Unknown
lHostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @aj??v?}g?i¾}?????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      @9      @A      @I      @aj??v?}g?i?v?} ???Unknown
[HostAddV2"Adam/add(1      @9      @A      @I      @a??T??"d?i<?~O:4???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a?????`?iX??E???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a?????`?it????U???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a?????`?i??X[?f???Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1      @9      @A      @I      @a?????`?i???	Yw???Unknown
?HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1      @9      @A      @I      @a?????`?iȀ?? ????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1      @9      @A      @I      @a?????`?i?q2g?????Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      @9      @A      @I      @a?????`?i c??????Unknown
[HostPow"
Adam/Pow_1(1      @9      @A      @I      @aU_c?Z?i????????Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @aU_c?Z?i`~3-?????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aU_c?Z?i???????Unknown
? HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aU_c?Z?i???Db????Unknown
i!HostCast"mean_squared_error/Cast(1      @9      @A      @I      @aU_c?Z?ip'H??????Unknown
v"HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1      @9      @A      @I      @a??T??"T?i?Q9?????Unknown
v#HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1      @9      @A      @I      @a??T??"T?i?{ҡ? ???Unknown
\$HostGreater"Greater(1      @9      @A      @I      @a??T??"T?i<??
???Unknown
e%Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a??T??"T?i??\s???Unknown?
`&HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a??T??"T?i??!?%???Unknown
?'HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??T??"T?i%?D7)???Unknown
?(HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      @9      @A      @I      @a??T??"T?iLO??H3???Unknown
q)Host_FusedMatMul"sequential/dense_1/Relu(1      @9      @A      @I      @a??T??"T?i?yqZ=???Unknown
t*HostReadVariableOp"Adam/Cast/ReadVariableOp(1       @9       @A       @I       @aU_c?J?ih@J\D???Unknown
]+HostCast"Adam/Cast_1(1       @9       @A       @I       @aU_c?J?i@#??J???Unknown
t,HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @aU_c?J?i???|Q???Unknown
v-HostAssignAddVariableOp"AssignAddVariableOp_2(1       @9       @A       @I       @aU_c?J?i???-3X???Unknown
X.HostCast"Cast_3(1       @9       @A       @I       @aU_c?J?i?[?s?^???Unknown
V/HostMean"Mean(1       @9       @A       @I       @aU_c?J?i?"???e???Unknown
u0HostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @aU_c?J?ix?^?Ul???Unknown
b1HostDivNoNan"div_no_nan_1(1       @9       @A       @I       @aU_c?J?iP?7Es???Unknown
?2HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1       @9       @A       @I       @aU_c?J?i(w??y???Unknown
}3HostMaximum"(gradient_tape/mean_squared_error/Maximum(1       @9       @A       @I       @aU_c?J?i >??x????Unknown
}4HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1       @9       @A       @I       @aU_c?J?i??/????Unknown
?5HostSquaredDifference"$mean_squared_error/SquaredDifference(1       @9       @A       @I       @aU_c?J?i?˚\?????Unknown
u6HostSum"$mean_squared_error/weighted_loss/Sum(1       @9       @A       @I       @aU_c?J?i??s??????Unknown
?7HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1       @9       @A       @I       @aU_c?J?i`YL?Q????Unknown
~8HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1      ??9      ??A      ??I      ??aU_c?:?i̼8?????Unknown
Y9HostPow"Adam/Pow(1      ??9      ??A      ??I      ??aU_c?:?i8 %.????Unknown
o:HostReadVariableOp"Adam/ReadVariableOp(1      ??9      ??A      ??I      ??aU_c?:?i??Qc????Unknown
v;HostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??aU_c?:?i??s?????Unknown
v<HostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??aU_c?:?i|J??????Unknown
v=HostAssignAddVariableOp"AssignAddVariableOp_4(1      ??9      ??A      ??I      ??aU_c?:?i??ֹt????Unknown
V>HostCast"Cast(1      ??9      ??A      ??I      ??aU_c?:?iT??ϲ???Unknown
X?HostCast"Cast_4(1      ??9      ??A      ??I      ??aU_c?:?i?t??*????Unknown
X@HostCast"Cast_5(1      ??9      ??A      ??I      ??aU_c?:?i,؛"?????Unknown
XAHostEqual"Equal(1      ??9      ??A      ??I      ??aU_c?:?i?;?E?????Unknown
TBHostMul"Mul(1      ??9      ??A      ??I      ??aU_c?:?i?th<????Unknown
wCHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??aU_c?:?ipa??????Unknown
yDHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??aU_c?:?i?eM??????Unknown
wEHostCast"%gradient_tape/mean_squared_error/Cast(1      ??9      ??A      ??I      ??aU_c?:?iH?9?M????Unknown
uFHostMul"$gradient_tape/mean_squared_error/Mul(1      ??9      ??A      ??I      ??aU_c?:?i?,&??????Unknown
uGHostSum"$gradient_tape/mean_squared_error/Sum(1      ??9      ??A      ??I      ??aU_c?:?i ?????Unknown
HHostFloorDiv")gradient_tape/mean_squared_error/floordiv(1      ??9      ??A      ??I      ??aU_c?:?i???9_????Unknown
uIHostSub"$gradient_tape/mean_squared_error/sub(1      ??9      ??A      ??I      ??aU_c?:?i?V?\?????Unknown
}JHostRealDiv"(gradient_tape/mean_squared_error/truediv(1      ??9      ??A      ??I      ??aU_c?:?id??????Unknown
?KHostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1      ??9      ??A      ??I      ??aU_c?:?i?Ģp????Unknown
}LHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1      ??9      ??A      ??I      ??aU_c?:?i<????????Unknown
MHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1      ??9      ??A      ??I      ??aU_c?:?i????&????Unknown
?NHostSigmoidGrad"4gradient_tape/sequential/dense_2/Sigmoid/SigmoidGrad(1      ??9      ??A      ??I      ??aU_c?:?iH??????Unknown
iOHostMean"mean_squared_error/Mean(1      ??9      ??A      ??I      ??aU_c?:?i??u.?????Unknown
?PHostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??aU_c?:?i?bQ8????Unknown
|QHostDivNoNan"&mean_squared_error/weighted_loss/value(1      ??9      ??A      ??I      ??aU_c?:?iXrNt?????Unknown
?RHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??aU_c?:?i??:??????Unknown
?SHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??aU_c?:?i09'?I????Unknown
?THostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??aU_c?:?i??ݤ????Unknown
oUHostSigmoid"sequential/dense_2/Sigmoid(1      ??9      ??A      ??I      ??aU_c?:?i     ???Unknown
4VHostIdentity"Identity(i     ???Unknown?
JWHostReadVariableOp"div_no_nan_1/ReadVariableOp(i     ???Unknown
JXHostMul"&gradient_tape/mean_squared_error/mul_1(i     ???Unknown*?O
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?P@9     ?P@A     ?P@I     ?P@a??W????i??W?????Unknown?
dHostDataset"Iterator::Model(1      @@9      @@A      5@I      5@aa???{??i۶m۶m???Unknown
oHost_FusedMatMul"sequential/dense/Relu(1      1@9      1@A      1@I      1@aq2?<p??i?)???d???Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      .@9      .@A      .@I      .@a??wÏ???iO#,?4????Unknown
iHostWriteSummary"WriteSummary(1      *@9      *@A      *@I      *@a?)???d??i??W?????Unknown?
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      *@9      *@A      *@I      *@a?)???d??i?m۶m????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1      &@9      &@A      &@I      &@aZtl???im?~T????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1      &@9      &@A      &@I      &@aZtl???i#\E;S???Unknown
s	HostDataset"Iterator::Model::ParallelMapV2(1      &@9      &@A      &@I      &@aZtl???i?_?"???Unknown
?
HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1      $@9      $@A      $@I      $@aigJ??8??iP??????Unknown
?HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      $@9      $@A      $@I      $@aigJ??8??iǬ?:6???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      "@9      "@A      "@I      "@axÏ????iT???P???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1      "@9      "@A      "@I      "@axÏ????i???/N???Unknown
gHostStridedSlice"strided_slice(1      "@9      "@A      "@I      "@axÏ????i?P@??????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1       @9       @A       @I       @a??%f-??i??n?Q]???Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @a??%f-??i???)?????Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1       @9       @A       @I       @a??%f-??i?K?Z(????Unknown
lHostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @a?{a????i??%f-???Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      @9      @A      @I      @a?{a????ie????????Unknown
[HostAddV2"Adam/add(1      @9      @A      @I      @aJ??8D??i"??U?3???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @aigJ??8??i?~T?????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @aigJ??8??i^E;Sz????Unknown
VHostSum"Sum_2(1      @9      @A      @I      @aigJ??8??i?n?Q]b???Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1      @9      @A      @I      @aigJ??8??i???P@????Unknown
?HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1      @9      @A      @I      @aigJ??8??i8?rO#,???Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1      @9      @A      @I      @aigJ??8??i??/N????Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      @9      @A      @I      @aigJ??8??it?L?????Unknown
[HostPow"
Adam/Pow_1(1      @9      @A      @I      @a??%f-??i?i???F???Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9       @A      @I       @a??%f-??ip?~T????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??%f-??i??
????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??%f-??ilgJ??8???Unknown
i HostCast"mean_squared_error/Cast(1      @9      @A      @I      @a??%f-??i???Gu????Unknown
v!HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1      @9      @A      @I      @aJ??8D~?iI;Sz?????Unknown
v"HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1      @9      @A      @I      @aJ??8D~?i??Ĭ????Unknown
\#HostGreater"Greater(1      @9      @A      @I      @aJ??8D~?i:6?????Unknown
e$Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @aJ??8D~?if???{???Unknown?
`%HostDivNoNan"
div_no_nan(1      @9      @A      @I      @aJ??8D~?i?8D????Unknown
?&HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aJ??8D~?i$??v?????Unknown
?'HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      @9      @A      @I      @aJ??8D~?i?7??.1???Unknown
q(Host_FusedMatMul"sequential/dense_1/Relu(1      @9      @A      @I      @aJ??8D~?i??m۶m???Unknown
t)HostReadVariableOp"Adam/Cast/ReadVariableOp(1       @9       @A       @I       @a??%f-t?i!a??????Unknown
]*HostCast"Adam/Cast_1(1       @9       @A       @I       @a??%f-t?i`tl????Unknown
t+HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a??%f-t?i??P@?????Unknown
v,HostAssignAddVariableOp"AssignAddVariableOp_2(1       @9       @A       @I       @a??%f-t?i?_?"???Unknown
X-HostCast"Cast_3(1       @9       @A       @I       @a??%f-t?i
??|7???Unknown
V.HostMean"Mean(1       @9       @A       @I       @a??%f-t?i\?3??_???Unknown
u/HostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a??%f-t?i?^q2????Unknown
b0HostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a??%f-t?i??=?????Unknown
?1HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1       @9       @A       @I       @a??%f-t?i?
?????Unknown
}2HostMaximum"(gradient_tape/mean_squared_error/Maximum(1       @9       @A       @I       @a??%f-t?iX]b?B???Unknown
}3HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1       @9       @A       @I       @a??%f-t?i????)???Unknown
?4HostSquaredDifference"$mean_squared_error/SquaredDifference(1       @9       @A       @I       @a??%f-t?iֱ?n?Q???Unknown
u5HostSum"$mean_squared_error/weighted_loss/Sum(1       @9       @A       @I       @a??%f-t?i\E;Sz???Unknown
?6HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a??%f-t?iT??????Unknown
~7HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1      ??9      ??A      ??I      ??a??%f-d?it۶m۶???Unknown
Y8HostPow"Adam/Pow(1      ??9      ??A      ??I      ??a??%f-d?i????????Unknown
o9HostReadVariableOp"Adam/ReadVariableOp(1      ??9      ??A      ??I      ??a??%f-d?i??:6????Unknown
v:HostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a??%f-d?i?Z(?c????Unknown
v;HostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a??%f-d?i?/N????Unknown
v<HostAssignAddVariableOp"AssignAddVariableOp_4(1      ??9      ??A      ??I      ??a??%f-d?itl????Unknown
V=HostCast"Cast(1      ??9      ??A      ??I      ??a??%f-d?i4ڙ??/???Unknown
X>HostCast"Cast_4(1      ??9      ??A      ??I      ??a??%f-d?iT??8D???Unknown
X?HostCast"Cast_5(1      ??9      ??A      ??I      ??a??%f-d?it???FX???Unknown
X@HostEqual"Equal(1      ??9      ??A      ??I      ??a??%f-d?i?Ytl???Unknown
TAHostMul"Mul(1      ??9      ??A      ??I      ??a??%f-d?i?.1k?????Unknown
wBHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??%f-d?i?W?Δ???Unknown
yCHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??%f-d?i??|7?????Unknown
wDHostCast"%gradient_tape/mean_squared_error/Cast(1      ??9      ??A      ??I      ??a??%f-d?i???)????Unknown
uEHostMul"$gradient_tape/mean_squared_error/Mul(1      ??9      ??A      ??I      ??a??%f-d?i4??W????Unknown
uFHostSum"$gradient_tape/mean_squared_error/Sum(1      ??9      ??A      ??I      ??a??%f-d?iTX?i?????Unknown
GHostFloorDiv")gradient_tape/mean_squared_error/floordiv(1      ??9      ??A      ??I      ??a??%f-d?it-б????Unknown
uHHostSub"$gradient_tape/mean_squared_error/sub(1      ??9      ??A      ??I      ??a??%f-d?i?:6????Unknown
}IHostRealDiv"(gradient_tape/mean_squared_error/truediv(1      ??9      ??A      ??I      ??a??%f-d?i??_?"???Unknown
?JHostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1      ??9      ??A      ??I      ??a??%f-d?iԬ?:6???Unknown
}KHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1      ??9      ??A      ??I      ??a??%f-d?i???hgJ???Unknown
LHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1      ??9      ??A      ??I      ??a??%f-d?iW?Δ^???Unknown
?MHostSigmoidGrad"4gradient_tape/sequential/dense_2/Sigmoid/SigmoidGrad(1      ??9      ??A      ??I      ??a??%f-d?i4,?4?r???Unknown
iNHostMean"mean_squared_error/Mean(1      ??9      ??A      ??I      ??a??%f-d?iT??????Unknown
?OHostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a??%f-d?it?B????Unknown
|PHostDivNoNan"&mean_squared_error/weighted_loss/value(1      ??9      ??A      ??I      ??a??%f-d?i??hgJ????Unknown
?QHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a??%f-d?i????w????Unknown
?RHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a??%f-d?i?U?3?????Unknown
?SHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a??%f-d?i?*ڙ?????Unknown
oTHostSigmoid"sequential/dense_2/Sigmoid(1      ??9      ??A      ??I      ??a??%f-d?i
     ???Unknown
4UHostIdentity"Identity(i
     ???Unknown?
JVHostReadVariableOp"div_no_nan_1/ReadVariableOp(i
     ???Unknown
JWHostMul"&gradient_tape/mean_squared_error/mul_1(i
     ???Unknown