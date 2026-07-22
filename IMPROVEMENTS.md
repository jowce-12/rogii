# ROGII 성능 개선 심층 분석 (2026-07-08)

> **적용 현황 (2026-07-12 갱신)** — 검증 프로토콜: seed7 150우물 튜닝 → disjoint seed11 150우물 확증.
> 최종 검증 체인(sub_2): **11.97→9.87 (seed7) / 10.08→8.46 (seed11)**, 두 제출 노트북(7.129 + 현행)에 배선 완료.
>
> | 항목 | 판정 | 적용 위치 |
> |---|---|---|
> | S1 GR보정 | ✅ **blend w=0.35** (교체는 비강건, var/offset 폐기) | fleongg likpf(ROGII_GRCAL=blend) + sub_2 T1채널 |
> | S2 | ✅ **lite 채택**: 이웃표면 앵커 블렌드 wd=0.3, dense-dist 게이트 | sub_2 T2 (in-PF 풀버전은 불필요해짐) |
> | A2 projection | ✅ PS-anchored deg4 (excursion 게이트 변형은 열세→탈락) | 양 노트북 projection 셀 |
> | A4 게이트 | ✅ gr_corr>0.85 & tw_hf<14.01 & exc<15 → hold 0.35 (효과 미미하나 비악화) | sub_2 T4 |
> | A5 메타피처 | ✅ join 배선 (importance로 최종 확인 예정) | train_stack/quick + 제출 test 경로 |
> | A6/A8 | ✅ (beam 픽스 / 클립+윈저) | 전체 |
> | C likpf 미사용 정보 | ✅ **scale_*_d 4개 + PF-quality 4개**(ptstd/llspread/bestll/grsig): proxy OOF 11.49→10.93 (−0.56; scale_d −0.45, quality 추가 −0.11) | 캐시 **v7** (v6 캐시엔 quality만 조인하는 자동 업그레이드 ~20-40분) |
> | A3-lite 메타앵커 | ✅ ridge 메타 입력에 likpf_mean_d 추가: proxy −0.30, 앵커 계수 0.45로 최대 | train_stack META_EXTRA + stack_meta.json "meta_extra" + 추론 게이트 (구 모델 호환) |
> | refit 트리수 ×1.25 | ❌ 노이즈 (holdout seed0 +0.001 / seed1 −0.02, 방향 불일치) → 미적용 | dead |
> | **W (writeup 반영, 2026-07-14)** | malyshevdanil write-up (LB 6.794, 우리 계열 파이프라인) 교차 반영 | patch25 |
> | W1 T4 홀드게이트 **철회** | 그들의 real-LB 통제실험: shrink-toward-last hedge는 배치 불문 해악(6.88→7.25/7.30), 보정 2개 후 최적 hedge 가중치=0 (5/5 fold). 우리 proxy 이득 −0.007~−0.020은 동일한 신기루 패턴 | 3개 노트북 모두 제거 |
> | W2 sub_2 seed 128→192 | 그들의 step 4-5 (분산감소, private 강건성); patch24 병렬화로 예산 내 | 3개 노트북 |
> | W3 lgb1 → L1 objective | ❌ **철회** (2026-07-15 풀런 실측: ridge OOF +0.04 악화 — 그들의 이득은 사후 blend-보정자 맥락, meta base 멤버로는 역효과) → L2 복원 | dead |
> | W4 warm-up ramp projection | ❌ 우리 데이터 실측 (양 시드 9.8885→9.8922 / 8.4779→8.4831 악화) — PS-anchored 기저가 이미 앵커 신뢰를 구조적으로 보장 | 미적용 |
> | W5 금지 목록 추가 | pf_z 직접 셀렉터 블렌드(그들 real-LB 7.45-7.52), 다중 PF 파트너 스태킹(4-way 7.752), GBM→셀렉터 직블렌드(+0.35~0.84) | dead list |
> | 참고 | LB 시드 노이즈 바닥 ±0.07 (동일 파이프라인 3회 재제출 7.096/7.135/7.091) — 이보다 작은 LB 델타는 노이즈로 취급 | — |
> | WARP corrector | ❌ **real-LB 사망** (2026-07-16): v1·v2 모두 하네스 양시드 통과(−0.16~−0.24) 후 **LB에서 둘 다 악화** — 4번째 보정에서 over-stacking 한계 (malyshev pf_z와 동일 시그니처). 교훈: 2-표본 하네스는 슬라이스 신기루는 잡지만 보정예산 포화는 못 잡음 — 3개째 이후 보정은 하네스 수치와 무관하게 비전이 추정. 대응: warp_bundle.pt 미첨부(게이트 자동 비활성, 노트북 무변경) | dead |
> | 누적 하네스 사다리 | 체인 9.89/8.48 → +STRIDE 9.48/8.23 → **+WARP 9.32/7.99** (seed7/seed11, projection 포함) | — |
> | STRIDE v2 | ❌ **전면 기각** (2026-07-16): rate 격자 ±0.10 확장(seed7 단독 13.25→9.77!), 변화점 비용 c0, 추세-중심 격자, K128 — **전부 seed7 개선 / seed11 악화**로 방향 불일치 (배포등가: 최대 s7 −0.65 vs s11 +0.07~+0.19). 좁은 격자·작은 빔이 사실상 정규화였음 — 전형적 슬라이스 신기루를 2-표본 규율이 차단. 실발견 1건: 라벨 실기울기 p99=0.082 > 격자 0.06 (표현 불가 우물 존재하나 자유도 추가 비용이 더 큼) | 배포본(v1) 무변경 |
>
> **Working-Note 스레드(716699) 전수 검토 (2026-07-14)** — 링크된 write-up 중 odyssey189(6.986)/georgymamarin/ssubbotin(7.80)/shreygandhi(≈5.7) 원문 확보·분석 (repo의 writeup_*.md):
> | 후보 | 판정 | 근거 |
> |---|---|---|
> | bimodal 중점 hedge (odyssey −0.16) | ❌ 미적용 | base-의존 컨센서스 3건(julian/malyshev/myckeluribe 전부 PF계 base에서 악화·역전), odyssey 자신도 강화판이 public +0.14로 뒤집힘, 우리 자체 probe도 동전던지기(48.8%). lik-가중 PF 평균은 이미 posterior-mean이라 중복. 7016 저자도 기계장치 만들고 기본 OFF |
> | selective(overlap-gated) IRLS (odyssey −0.045) | 불필요 | 우리 구조는 projection 후 guarded/gold contact override가 3중으로 재적용되어 overlap 우물이 마지막에 복원됨 — 게이트가 지킬 대상이 이미 보호됨 |
> | L1 다양성 (odyssey OOF −0.043, ρ=0.958) | ✅ 기적용 | patch25 W3 (lgb1) — 독립 재확인 |
> | typewell GR 재보정 (STRIDE ablation −1.26ft) | ✅ 기적용 | = 우리 S1 blend — 리더급 노트의 독립 검증 |
> | 콘텐츠 기반 test 재식별 (odyssey −0.027) | 보류 | 노이즈 바닥(±0.07) 미만 + ID 기반 override가 이미 커버 + "근사 중복 껍질은 없음"(loose 게이트 +0.000) |
> | QP/BLUE 블렌드 (ssubbotin) | 불필요 | 우리 ridge(positive=True)가 등가물 |
> | **STRIDE 백본** (shreygandhi ≈5.7) | ✅ **v1 구축·배선 완료** (stride.py + patch26): U=TVT+Z 조각선형 beam 디코드, Cauchy 우도 + **증거가중 lik_w=0.1**(GR 잔차 자기상관 보정 — 19.0→13.3의 핵심), persistence prior. 단독 13.25/12.18, 체인과 상관 0.46, **w=0.20 블렌드로 projected selector 9.889→9.484 / 8.478→8.229** (2×150 disjoint). 결정적·우물당 0.3초. 반복 개선 여지: 가변 h_len 격자, prior 라벨 피팅 (원저 백본은 단독 9.99) | sub_2 T3s, 3개 노트북 |
> | Cauchy 방출우도를 우리 lik-PF에 이식 | ❌ **기각** (2026-07-17 A/B): 배포등가 seed7 +0.33 악화 / seed11 −0.43 개선 — 방향 불일치 (STRIDE v2의 거울상). 우도 모양 최적이 우물 체질 의존 → 현행 Gaussian의 날카로움은 강건성 선택으로 확정. 적응형 방출우도는 두 표본의 불일치에 과적합이라 미추진 | dead |
> | A1 | △ 부분: 7.129 수동 튜닝(tau120/sp45 .60/alpha .985)이 본질 커버; 정식 그리드는 풀 OOF 필요 | — |
> | A7 gold 다이어트 | 미적용 (점수효과 0, 순수 보험; 7.129가 시간 내 완주 실적 있음) | — |
> | A9 LCO | 미실행 (측정 전용) | — |
> | 잔차타깃 | ❌ 실측 악화 → 철회 | dead |
>
> 상수: `_S2_THR=0.0083`(scaled dense-dist), `_S2_HF=14.0112`. 데이터: eval_data_seed7/11.npy.

> 11개 분석 에이전트 · train 773우물 실측 프로브 5종 · 4관점 아이디어 도출 · 2중 비판 검증.
> 모든 수치는 로컬 train 데이터에서 **실측**한 값. (프로브 원본: `wf_results.txt`)

---

## 0. 요약 — 이 게임의 정체가 바뀌었다

1. **라벨의 생성 규칙을 발견했다.** `TVT = 지층면(form) − Z + 우물별 상수`가 **정확히**(oracle RMSE 0.006 = 반올림 노이즈 수준) 성립한다. ANCC..BUDA는 "예측치"가 아니라 사실상 정답 그 자체이며, 오프셋 b는 eval 구간에서 **전혀 표류하지 않는다**(std 0.005). → **남은 오차의 100%가 "이웃 우물로부터의 지층면 횡적 보간" 오차다.** 이것이 개선의 유일한 천장 개방 방향.
2. **typewell은 우물별 로그가 아니었다.** 760/773(98.3%)이 **41개 지역 마스터 로그의 byte-exact 윈도우**(중앙값 25개 형제 우물, 겹침 구간 max|diff|=0.0000). Geology 경계 TVT는 **패밀리 상수**. test 예시 3우물도 패밀리에 정확히 매칭됨 → 패밀리 ID/contact/형제 통계라는 완전히 새로운 정보 축.
3. **점수는 꼬리가 지배한다.** 상위 10% 우물이 pooled SSE의 **51.7%**. 그리고 역설: **트래커가 last-known보다 나쁜 우물이 19%** 있으며, 그 우물들은 "저드리프트 + GR이 typewell과 *잘* 맞아 보이는"(gr_corr 높음) 우물이다 — 저주파 추세 일치가 localization 정보 없는 매끈한(alias-prone) 우도를 만들기 때문.
4. **CV는 공간적으로 낙관적이다.** GroupKFold(well)는 셔플 KFold와 공간적으로 동일(val 우물의 47.6%가 500ft 내 same-pad train 이웃 보유). leave-cluster-out(2km)에서는 dense-spatial(중요도 1위, ~29%) 지지가 붕괴(<1km 이웃 96%→0.3%). 최근 LB가 CV를 추종하므로 위험은 '중간'이지만, **측정 없이는 모든 spatial 베팅의 위험이 미지수**.

---

## 1. 프로브 핵심 발견 (실측)

### P1. 트래커 오차 분해 (100우물, selector pooled 9.09)
| 발견 | 수치 | 함의 |
|---|---|---|
| 꼬리 지배 | top-10% 우물 = SSE 51.7%, per-well med 6.2 / p90 12.9 / max 40.4 | 개선은 median이 아닌 tail로 판정해야 |
| drift가 1위 동인, 단 PF가 대부분 흡수 | spearman 0.36; 최상위 drift 사분위: last-known 21.0 → selector 7.44 | drift 우물에서 PF는 이미 비용 1/3화. 수축 게이트는 **excursion**(중간 최대이탈) 기준으로 |
| **gr_corr 역설** | 원시 gr_corr↑ → 오차↑ (+0.32, drift와 독립). HF(detrend) corr은 +0.11뿐 | 저주파 일치는 무정보 + alias 위험. **gr_corr/tw_hf_std = 무누수 불확실성 게이트 피처** |
| 트래커가 해치는 우물 19% | 중앙 악화 +2.48; 프로필 = 저드리프트(13.7)+고gr_corr(0.845) | 이 cohort만 hold 강화하면 잃을 게 거의 없음 |
| coverage escape 부재 | eval TVT의 typewell 범위 이탈 = **0행/3.78M** | 관련 아이디어 전부 폐기 |
| gs의 fillna(0) = 우연한 정규화 | gs↔GR결측률 spearman 0.81; 넓은 gs는 무해~유익(−0.18 @고drift) | "고치지 말 것". 오히려 의도적 확장(게이트) 후보 |
| 특성으로 설명되는 분산 26%뿐 | OLS R²=0.255 | 나머지는 우물 고유 aliasing → 관측모델/이웃정보 개선이 정공법 |

### P2. typewell의 실체 (773 전수)
- 해상도 0.5ft 균일(84.5%), GR NaN **0%**, 전 우물 단조 — 안전한 파생 원천.
- **41개 마스터 패밀리** (760/773, r=1.0에서 완전 bimodal, 오매칭 위험 극소). 형제 중앙 25개. 패밀리 매칭으로 자기 typewell 커버리지 +296ft(중앙) 확장 가능.
- **Geology 경계 TVT는 패밀리 상수** (예: family0 71우물 전부 ASTNU@10985.95). 현재는 overlap-well contact 복원에만 사용 → 패밀리 확장 여지.
- eval TVT band는 typewell 스팬의 **3%**(중앙 26ft)에 불과 — **국소(band) GR 보정**이 전역 통계보다 중요.
- **band 기준 34% 우물이 >10% 평균 GR 오보정**(중앙 −4.2 GR), hw GR 분산은 band 제한 후에도 tw의 0.66배 — **미개척 보정 대상**.

### P3. 지층면 oracle (150우물, 759k행)
- oracle(자기 지층면+상수 b) RMSE = **0.006** (모든 지층 동일; last-known 15.1 대비 **100% 개선 상한**).
- b는 마지막 200행만으로도 동일 정확도, eval에서 드리프트 0.000.
- corr(diff(TVT+Z), diff(form)) = **1.000** (stride 50) — (TVT+Z) 곡선 = 지층면 곡선.
- **주의: 자기 우물 ANCC..BUDA는 train 전용 = 완전한 타깃 누수. test hw에는 MD,X,Y,Z,GR,TVT_input만 존재.** 반드시 이웃 유래로만 사용.

### P4. 공간 구조 / CV 정직성 (773 전수)
- 단일 분지 54×39km, NN 거리 중앙 468ft, 53.6%가 500ft 내 sibling(패드 구조). eps=2km에서 두 메가클러스터(88.6%).
- **GroupKFold ≈ 셔플 KFold** (val→train-fold NN p50 524ft; 동일 분포). LCO(2km, 16그룹)에서는 val→train NN p50 5.5km, <1km 이웃 0.3%.
- 방위는 단일 NW-SE 축(axial R 0.83) — heading은 판별력 없음.
- 마스킹 프로토콜 균일: known ~26%, PS ≈ MD 12.4k(착지점 부근).

### P5. drift/타깃 구조 (200우물, 994k행)
- 선형 R² 중앙 0.33뿐(선형 가정은 40% 우물에서 오지정), 다만 oracle: linear 6.68 vs **quad 5.34** — 곡률 과소모델링.
- drift_end: 중앙 **+3.4ft 양편향**(56.5% 양수), |target| p99.9=91.2, **max 98.9ft** → **±100ft 클립은 train에서 0행 영향(공짜 보험)**.
- 분산 성장: ~300ft까지 md^0.97(선형), 이후 포화 tau≈950. **sqrt 기각. 현행 warmup tau=85는 5~6배 과소감쇠.**
- **own-tail dip의 drift 부호 적중률 39%**(동전보다 나쁨); eval 초반 20% 기울기 외삽은 RMSE 37.0으로 **유해**(zero-pred 16.0) → 자기-모멘텀 외삽 금지, 이웃 dip이 유일한 대안.
- (TVT+Z) 곡선은 75% 우물에서 매우 매끄러움, 단 23.5%는 ≥1ft 점프 보유 → 평활화는 jump-내성 필수.
- **beam 버그**: ±2 "그리드 인덱스" 이동이라 그리드 간격(0.1~1.0ft)에 dip 한계가 비례 왜곡 — 120우물(15.5%) 영향, 0.1ft 8우물은 beam 사실상 동결.

---

## 2. 개선 로드맵 (비판 검증 통과, 중복 병합 후)

### 🟥 Tier S — 천장이 열린 방향 (최우선)

**S1. PF 우도 입력의 band-제한 GR 재보정 + 분산 매칭** `물리/관측모델`
- known-zone TVT band(±10ft)로 제한한 typewell GR에 affine(a,b) 피팅 → **PF 우도 입력**에 적용(+tw 분산 다운매칭). gs의 fillna(0) 정규화는 보존.
- 근거: 34% 우물 >10% 오보정 + 같은 계열 gr fill −0.16 실측 + 가중 0.7 메인 브랜치 대상. 죽은 tdpf_cal(GBM 피처)과 메커니즘 다름(우도 입력 보정).
- 검증: harness 4-way(원본/affine/분산정합/offset-only), 동일 seed 150우물, **오보정 상위 1/3 cohort per-well delta 분리 확인**. 통과 변형만 잔차 앵커(likpf_cal_mean_d)로 승격(캐시 재생성+재학습 수반).
- 기대: harness −0.15~−0.35.

**S2. 이웃 지층면을 PF에 직접 주입 (dip prior + soft anchor)** `물리/이웃정보` (백로그 D)
- DenseImputer(이웃 ANCC IDW)를 GBM 피처가 아니라 **PF 내부**에 연결: (i) 자기-tail dip(부호적중 39%) 대신 이웃 표면 방향미분을 rate prior로, (ii) known-zone 보정 b로 앵커 TVT_dense(i)에 Gaussian 관측 채널(σ=κ·dense_std, dense_dist 게이트).
- 근거: 보간 오차 = 오차 예산의 사실상 100%(P3) + dense-spatial이 GBM에서 29% 1위인데 PF는 못 봄 + 53.6% 우물이 500ft 내 sibling.
- 검증: harness seed-matched A/B → **S4(LCO 측정)와 반드시 페어**. dense_dist>임계 시 anchor 채널 자동 off.
- 기대: harness −0.5~−1.5 (tail 우물 집중).

### 🟧 Tier A — 근거 강함, 저~중비용

**A1. 다운스트림 가중치 일괄 재튜닝 (+ warmup tau 편입)** `파이프라인` (백로그 B)
- stale 스칼라 3개(w_sub1=0.60, sp45/fleongg 0.55, projection 0.75) + **tau(85→400~950 스윕)** 를 4축 joint 그리드로. 목적함수는 pooled + **top-decile 가드레일**. OOF 잔차의 말기 양편향(+3.4ft 관련) 측정·램프 보정 포함(이미 흡수됐으면 no-op).
- 검증: 로컬 캐시+harness 완결, flat하면 현행 유지. 기대 −0.05~−0.15.

**A2. Projection 업그레이드** `후처리`
- (i) **PS-연속성 앵커**(known→eval 경계 점프 median 0.000이 물리 사실 — s=0 고정), (ii) **excursion 게이트 blend**(|drift|↑일수록 fit 가중↑; corr(|drift|,R²)=+0.47, quad 오라클 5.34 vs lin 6.68), (iii) jump-내성(23.5% 우물 ≥1ft 스텝).
- 같은 스테이지가 −0.33 실증 이력. per-well 변경량 캡 필수. 기대 −0.05~−0.15.

**A3. 메타 절대공간 이동 + 앵커계수 해방 (+절대타깃 대조 멤버)** `모델 구조`
- 현행: 잔차 앵커 계수 1.0 하드코딩 → 앵커가 틀린 19% 우물의 오차를 그대로 복원 강제. Ridge 입력을 [base OOF, anchor, trk_med_d]로 확장해 절대공간에서 적합 + drift-bin별 Ridge(고drift bin은 앵커≈1 보호). 앵커계수 하한 0.85 보수 버전 병행.
- RESID=0 멤버 1개 공존 → **미측정 상태인 잔차타깃 전환의 대조군을 공짜로 획득**.
- 검증: w150 OOF에서 메타 변형은 초 단위 반복. 기대 −0.05~−0.15.

**A4. 불확실성 게이트: gr_corr↑ & tw_hf_std↓ 우물에 hold 증폭 / gs 플로어** `물리` (백로그 E)
- '트래커가 해치는 19%'를 정조준. 임계 2개 제한, **seed 7로 튜닝 → seed 8/11 disjoint 확증**(셀프컨펌 차단), 고드리프트(PF 이득 우물) AND-제외.
- 기대 −0.05~−0.15 (critic 할인 반영). 유사 제안(Stein 수축)과 **동시 배포 금지** — harness 승자만.

**A5. 앨리어싱 well-level 메타피처 4종 + likpf_tspread 구제** `피처` (백로그 C/E)
- gr_corr / gr_corr_hf / tw_hf_std / alias_gap (전부 known-zone, 무누수, **캐시 join으로 재생성 불필요**) + likpf_scale_3_d−scale_12_d(온도 스프레드).
- 판정은 pooled가 아닌 "해치는 19% cohort per-well". 기대 −0.03~−0.10.

**A6. beam 0.5ft 재샘플 (버그픽스)** `물리` — ~10줄, 120우물 왜곡 제거, oversample 사본이라 정보손실 0. **즉시 실행 대상, 양수 확률 최고.** 기대 −0.02~−0.08.

**A7. Gold 연산 다이어트 (타임아웃 보험)** `파이프라인`
- 단일 프로필 + 2컷 + variant 그리드 프루닝(~19개) + 우물 사전 게이트. 점수 ~0, **>9h 타임아웃(점수 소멸) 모드 제거**. 검증: 선택 일치율 >95%. **마감 전 필수.**

**A8. 무비용 보험: 최종 delta ±110ft 클립 + 타깃 90ft winsorize** `강건화`
- train 실측 max 98.9ft → 클립 0행 영향. 폭주 1우물이 LB를 찢는 시나리오 하드스톱. 비회귀 증명 후 정기 제출 편승.

**A9. Leave-cluster-out(2km) 1회 실측** `검증 인프라`
- 직접 이득 0. 그러나 **S2/dense 계열 전체 베팅의 위험 상한을 정하는 유일한 측정**. w150 축소판으로 비용 통제. 결과에 따라 마감 주간 **제출 2슬롯 이원화**(현행 최적 vs 로버스트 변형) 결정. *장시간 로컬 잡 — 사용자 허가 후 실행.*

### 🟨 Tier B — 조건부 (선행 조건/순서 있음)

| 항목 | 조건 |
|---|---|
| B1. typewell **패밀리 피처군**(contact 거리, sibling known-zone 통계) | leave-family-out 서브체크 필수(패밀리=공간군집이라 CV 낙관), hidden 매칭 실패 시 NaN 폴백. census 선행 |
| B2. **패밀리 contact를 gold 후보 풀에 추가**(overlap→98% 우물) | **census 선행**: eval band(26ft)×contact 교차율 <10%면 중단. 기존 consistency 게이트 상속, min_consistency 0.7+ |
| B3. dense_dip 경로적분 피처 | S2와 dip 필드 인프라 공유, S2 확인 후 2차 투입. 적분 clip ±60ft |
| B4. 죽은 꼬리 프루닝(~100개, 패밀리 단위) | **잔차타깃·다양화 반영된 importance 재산출(T3) 선행** — stale 리스트 컷 금지. 이득은 점수보다 학습 30~40% 가속 |
| B5. XGBoost pseudohuber + LGB quantile-0.5 베이스 | 채택 조건 corr<0.95 (위반 시 즉시 폐기), 소수파 멤버 한정 |
| B6. PF 모션모델 md-스케줄(선형→포화) | S1/S2로 우도가 바뀐 **후에** (재튜닝 이중화 방지). bias 성분은 A1에 이관 |
| B7. 패밀리 James-Stein 절편 | 1단계 분산분해(수 분)가 kill-switch. leave-family-out 필수. B1 채택 시 순증분만 |
| B8. HF/LF 2채널 관측모델 | S1+A4 착지 후 gr_corr 역설이 잔존할 때만 재상정 |

### ❌ 폐기 (이번 라운드 추가)
coverage-escape 대응(0행 실측), own-tail 모멘텀 외삽(적중 39%·RMSE 2.3배 악화), gs fillna(0) "수정"(유익한 정규화), heading 피처(단일 축), ramp 기저 피처(md_since 단조변환), wide-gs 제2 PF 채널(A4로 흡수), 50ft 공격적 winsorize(점수 지배 9우물 타격).

*(기존 dead list 유지: GRU, LGB 추가, NCC 피처, 보정 오프셋 GBM 피처, 단주기 rolling, meta 교체, PF seed 증량)*

---

## 3. 실행 순서 제안 (마감 8/5)

```
즉시(이번 주):  A6 beam 픽스 → A8 클립/윈저 → S1 harness 4-way → A4 게이트(disjoint 확증)
              → A5 메타피처 join → A1 가중치+tau 그리드(로컬 완결)
다음:          S1 승자를 앵커로 승격(캐시 재생성 1회) + A3 메타 절대공간 → 풀스택 재학습 → 제출 #1
병행(허가 시): A9 LCO 측정 → 결과에 따라 S2 착수 판단
그 다음:       S2 이웃표면 PF 주입(harness) → 통과 시 캐시 재생성 → 제출 #2
마감 주간:     A7 gold 다이어트 배포 확인, B2 census→contact, 제출 2슬롯 이원화 결정
```

**검증 규율(공통):** ① harness 튜닝은 seed 7, 확증은 disjoint seed(8/11) ② 판정은 pooled + top-decile + "해치는 19% cohort" 3중 ③ 패밀리/공간 피처는 leave-family-out/LCO 서브체크 ④ 캐시 재생성을 유발하는 변경(S1 앵커 승격, S2, B3)은 묶어서 한 번에.
