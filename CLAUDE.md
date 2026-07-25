# ROGII Wellbore Geology Prediction — 캠페인 핸드오프 (2026-07-23 기준)

Kaggle 대회: 수평정 eval 구간의 TVT 예측, pooled RMSE. **마감 8/5, freeze 8/1, 제출 9시간 한도.**
LB 최고 기록 **6.663** (dip pole 시절). public 1위 6.568 (ultimate 포크).

## 현재 상태 (가장 중요)

- **노트북**: `rogii-geology-aware-ensembling-lb-7-129.ipynb` (+conservative 쌍둥이, 항상 동일 패치) — **patch46+47+48+49+50 적용 상태**
- **첨부**: GRU 데이터셋 = dip 25개 (`gru_fold{0-4}_{da,db,dc,dd,de}.pt`), fleongg 아티팩트 = `local-train`(stride v2, 8.0576), sub1 슬림 = ravii-retuning
- **제출 대기 독해**: ① patch48+49 (하네스 7.0962/5.8095), ② patch50 버전 (6.9983/5.7412). 기준점 6.663
- **하네스 사다리 다음 랭**: 일반 티어 0.10/0.40/0.50 (6.9112/5.6867, 엣지 미발견) — LB가 앞 랭을 확인해줄 때만 진행
- **freeze 계획**: 확인된 최고 랭 + 6.663 헤지(Kaggle 버전 히스토리 보존)

## 최종 블렌드 구조 (배포)

```
일반 우물:  0.15·sp45(0.3·sub1+0.7·selector) + 0.45·fleongg + 0.40·GRU(dip5)   [patch50]
몬스터 우물(risk=likpf_ptstd 우물평균 ≥3.39): 0.20/0.40/0.40                    [patch49]
→ (1−0.10)·blend + 0.10·STRIDE-v3                                              [patch48]
→ 몬스터 우물만 γ=1.09 진폭 (last + 1.09·(b−last))                              [patch40]
→ guarded 컨택트 오버라이드(오버랩 우물) → gold visible-prefix (conservative — cell 2 env가 하드코딩, 전 기간 conservative였음)
```

- GRU pole: dip 5-leg (_da 42/.25, _db 202/.25, _dc 777/.4, _dd 1301/.3, _de 555/.35), 31채널, dip 헤드, 융합 λ=4096, **추론은 CPU 고정**(map_location='cpu', .to() 없음 — 의도됨, 아래 참고)
- STRIDE-v3: 가변길이 격자 DP (stride3.py; wlen 0.5, sigp 0.012, 추세 초기화, 상수는 노트북 cell 24에 임베드)

## 반드시 아는 사실 (돈 주고 배운 것들)

1. **GPU forward는 이 GRU를 ~1.7ft 손상시킴** (isic 4060Ti에서 실측; 재귀 누적 저정밀). 모든 판정 아티팩트는 CPU로 계산할 것. Kaggle 프리뷰의 `[gru] checksum mean=` 라인과 로컬 CPU 재계산이 소수 4자리 일치 확인됨.
2. **spatial(이웃 표면) GRU 채널은 LB에서 실패** (7.140). LOCO(2km 제외) 평가로 사인 규명: 풀뱅크 5.9/4.8 → LOCO 9.5/9.8 (dip5 6.5/5.3보다 나쁨) — 하네스 이득 전부가 같은 패드 이웃 부풀림. **하네스는 spatial 계열 비교에선 실격**, 비-spatial 축에선 신뢰 이력 유효.
3. **gold 프로파일은 처음부터 conservative** (cell 2 env 하드코딩) — "balanced가 낫다"는 옛 결론은 라벨 착오로 무효. 히든 우물 gold는 베이스가 강할수록 마이너스 (conservative조차 신베이스에서 +0.05).
4. **padded-eval/단독 OOF 판독은 배신함** — 공식 판정은 클린 재계산(무패딩 CPU) → fleongg 상관 → **양시드(7/11, 2×150우물) 블렌드 게이트**. 튜닝은 seed7, 확인은 seed11.
5. **한 제출에 한 델타** — patch42(gold 풀 확장)를 검증 없이 LB 직행시켜 슬롯 태운 전례. 프리뷰 확인 라인 목록을 매 제출 전 대조.
6. LB 노이즈 ±0.07. 하네스→LB 전이율은 비-spatial 축에서 대략 50~100%.

## 정정: 아티팩트 오염 사고 (2026-07-25) — 반드시 읽을 것

`gru_oof_dipfused_ext.parquet`(내 CPU 산출물)이 사용자의 GPU 재실행에 **조용히 덮어써졌고**(같은 공유 디렉토리), 로그에는 CPU 숫자가 남아 있어 발견이 늦었다. 그 결과 "dip5 채택(−0.67/−0.45)"은 **CPU dip5 vs GPU dip3** 비교였고 대부분이 GPU forward 손상(~1.7ft)이었다. 파일은 `*.GPU_CONTAMINATED.parquet`로 격리.

**정직한 CPU-only 사다리 (양시드)**:
| 단계 | seed7 | seed11 |
|---|---|---|
| LB 6.663 구성 (CPU dip3 lam1024, .20/.50/.30, γ) | 7.3417 | 6.0389 |
| + dip5 pole | 7.3586 | 6.0188 (시드 스플릿) |
| + STRIDE-v3 pole 0.10 | 7.1980 | 5.9541 |
| + 계층 가중 (patch49+50) | 6.9983 | 5.7412 |

- **dip 5-leg는 게이트 탈락**(현행 구성에서 dip3 6.9610/5.7572 vs dip5 6.9983/5.7412). 15 vs 25 ckpt 차이는 노이즈(±0.02) → 재업로드 불필요하나 **leg 추가는 금지**(dip7 러너 WITHDRAWN 표기). leg가 늘수록 fleongg 상관은 **상승**(0.7357→0.7459).
- **진짜 이득 = v3 pole + 계층 가중: −0.38/−0.28** (내가 보고했던 −0.93/−0.66 아님).
- **규칙**: 판정 아티팩트는 사용 직전 **재채점으로 출처 검증**. 로그만 믿지 말 것. GPU 실행이 공유 디렉토리의 CPU 산출물을 덮어쓸 수 있다.

## fleongg 심층 세션 결과 (2026-07-25) — 이 모델은 국소 최적

- **메타 계층 종결**: 배포 `Ridge(1.66, positive, +likpf_mean_d)`가 6개 변형을 fleongg 단독·블렌드 모두에서 이김 (단독 8.0401 vs 티어드 8.0621 / 품질피처 8.0707 / LGBM 8.79; 블렌드 6.9983·5.7412 vs 전부 열위). `meta_lab.py`
- **베이스 모델 종결**: 150우물 프록시 기준선 6.8736 vs huber 7.00 / rate-space 7.11 / 거리 티어드 7.02 / capacity↑ 6.96 / capacity↓ 6.89 — 전부 악화. `measure_base.py`
- **γ 종결**: 일반 티어 γ(s11 전 구간 악화), 거리 램프(양시드 악화) → 1.09 몬스터 전용 유지. `gamma_lab.log`
- **진단**: fleongg는 GRU pole 대비 전 구간 열위이고 **컷에서 멀수록 격차 확대**(eval 길이 Q1→Q4 1.33→1.70ft, 공간 고립 Q4 2.39ft) = 행 단위 GBM의 장거리 외삽 한계
- **미결 카드**: STRIDE-v3를 fleongg 피처로 (`measure_v3feat.py` + `s3_all.parquet`) — 통과 시에만 fleongg 재학습 가치

## test TVT 온라인 활용 (transductive) — 3각도 측정 결과

1. **원리 확인**: 우물별 편향 지속성 실재 (앞/뒤 절반 corr 0.657, 보정 시 9.81→8.02). 단 거리 따라 상관 감쇠(0.67→0.35), 편향 크기는 증가(2.25→6.71ft) = 누적 언더드리프트 = γ가 이미 먹는 축
2. **값싼 자기보정 기각**: 프리픽스 인위 컷 + v3 백테스트 편향 보정 → s7 corr +0.208 / s11 −0.092 시드 스플릿. `prefix_selfcal.py`
3. **교차 우물 뱅크 확장 기각**: 테스트 프리픽스를 이웃 표면 뱅크에 투입 → 원시 TVT+Z 100.7→101.7(우물 상수 오염), ANCC 보정 스케일 31.8→37.0(개선 27.5%). 한 우물의 선형 밀집 샘플이 k=20 이웃 독점 + 기존 뱅크 국소 오차 되먹임. `testbank_probe.py`, `testbank2.log`
4. **유일한 잔여 경로**: fleongg 자신을 인위 컷에서 재예측(피처 재빌드 +1.8h 런타임) → 기대 −0.05~−0.09, freeze 전 수지 불합격

## 기각된 축 (재시도 금지, 근거는 메모리/로그)

spatial·spatial2 GRU 채널(LB+LOCO), fleongg 피처 추가 전반(mmprof 무시/xtrk 유해/grshape 취약 — likpf quality가 이미 커버), branch midpoint hedge(시드스플릿), c40 컷 제거(+0.15), hard-well 가중(+0.14, 몬스터 우물조차 악화), 채널 드롭아웃(클린에서 역전), 6-leg tvt(상관 상승), heel blend, 마스터 로그 확장(수혜 모집단 0/276), σ-temper ×1.3(실효 selector 비중 14%라 기대 −0.006, 보류), balanced/aggressive gold(히든 우물 유해).

## 작업 규칙 (사용자 지시로 굳어진 것)

- 사용자 프로세스/학습을 임의로 시작·중단하지 말 것. 질문에는 답만.
- 결정은 코드에 하드코딩 (env-var 숙제 금지). 학습↔제출 짝 변경은 동시에.
- 참조 스펙 구현은 처음부터 풀스펙으로 (단계 쪼개기 금지).
- 실측과 추정을 구분해 말하기. 유리한 해석 금지. 결과는 디스크 파일로 검증 (로그 붙임 의존 금지).
- 문제 발생 시 내 코드부터 의심. GPU 학습은 사용자가 직접 돌림.
- 노트북 패치는 반드시 patchNN.py 스크립트로 (--check/--undo 포함), 두 노트북 동일 적용, ast 검증.

## 핵심 파일 지도

- 판정 기계: `blend_eval.py`(selector_preds/OOF), `offline_tests.py`(pooled), `gru_fusion_mixed.py`(--tags 클린 재계산), `gru_fusion_dip5.py`, `stride3.py`(+`seg_prior.json`), `s3_preds_tuned.parquet`, `gru_oof_dipfused5.parquet`(현행 pole OOF), `loco_eval.py`(spatial 부검)
- gold 하네스: `stage2_gold_pool.py`+`stage2_worker.py`(컨택트 제외 히든 레짐; 현재 v3-gold conservative 실험 상태)
- 트레이너: `train_gru2.py`(--spatial/--spatial2/--dip/--wellw/--chdrop/--dropcuts, 기본값이 재현성 보장), 러너 `gru_ensemble_*.py`
- 패치 기록: `patch42.py`~`patch50.py` (42=기각·제거됨, 43+46=GRU/spatial 배선+경로수정, 44=제거됨, 45=구버전 가중 프로브, 47=롤백, 48=v3+dip5+λ4096+체크섬, 49=계층 가중, 50=일반 티어 N2)
- 검증: `parity_gru_dip.py`/`parity_gru_spatial.py`, `checksum_verify.log`
- 심층 사실: `IMPROVEMENTS.md`, `writeup_*.md`(수상작), `wiggle_writeup.md`

## 프리뷰 체크라인 (현행 버전 기준)

`[gru] 25 checkpoints loaded (25 dip-head, 0 spatial)` / `[gru] checksum mean=11906.4454`(3우물 프리뷰 기준) / 우물별 `STRIDE-v3 OK` / `[tier] monster mix 0.20/0.40/0.40 on N rows` / `[s3] stride-v3 pole mixed at 0.10` / `[3way] ... weights 0.15/0.45/0.40 normal-tier (N2)` / `[amp] gamma=1.09` / gold profile conservative. 프리뷰 sha256이 버전 간 동일한 것은 정상(오버랩 3우물은 컨택트 오버라이드가 전 행을 덮음).
