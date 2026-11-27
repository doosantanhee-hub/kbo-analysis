import streamlit as st
import pandas as pd
import numpy as np

# 페이지 기본 설정
st.set_page_config(
    page_title="야구단 전력 & 풀 리그 시뮬레이터",
    page_icon="⚾",
    layout="wide"
)

# ==========================================
# 1. 실제 리그 통계 상수 (사용자 제공 데이터 반영)
# ==========================================
LEAGUE_RPG = 5.07          # 리그 평균 경기당 득점(R/G) 및 실점(RA/G)
TEAM_RS_MAX = 5.73         # 팀 최다 경기당 득점 (공격 100점 기준)
TEAM_RS_MIN = 4.03         # 팀 최소 경기당 득점 (공격 0점 기준)
TEAM_RA_MIN = 3.85         # 팀 최소 경기당 실점 (수비/투수 100점 기준)
TEAM_RA_MAX = 5.80         # 팀 최다 경기당 실점 (수비/투수 0점 기준)
POSTSEASON_CUTOFF = 0.520  # 포스트시즌(5위) 진출 커트라인 승률
HOME_ADVANTAGE = 0.0       # 홈 어드밴티지 없음
PYTHAG_EXP = round((LEAGUE_RPG * 2) ** 0.287, 3)  # 역산 피타고리안 지수: 1.944

# 세션 상태 초기화
default_stats = {
    "starter": 65,
    "bullpen": 60,
    "batting": 64,
    "defense": 58,
    "baserunning": 55,
}
for key, val in default_stats.items():
    if key not in st.session_state:
        st.session_state[key] = val

def set_preset(starter, bullpen, batting, defense, baserunning):
    st.session_state.starter = starter
    st.session_state.bullpen = bullpen
    st.session_state.batting = batting
    st.session_state.defense = defense
    st.session_state.baserunning = baserunning

# ==========================================
# 2. 전력 점수 -> 득실점, 기대 승률, Log5 계산 함수
# ==========================================
def calc_team_metrics(starter, bullpen, batting, defense, baserunning):
    total_score = (
        starter * 0.30 +
        batting * 0.25 +
        bullpen * 0.20 +
        defense * 0.15 +
        baserunning * 0.10
    )
    
    offense_score = (batting * 25 + baserunning * 10) / 35.0
    defense_score = (starter * 30 + bullpen * 20 + defense * 15) / 65.0
    
    # 50점(5.07) 기준 비대칭 선형 보간
    if offense_score >= 50:
        rs_pg = LEAGUE_RPG + ((offense_score - 50.0) / 50.0) * (TEAM_RS_MAX - LEAGUE_RPG)
    else:
        rs_pg = LEAGUE_RPG - ((50.0 - offense_score) / 50.0) * (LEAGUE_RPG - TEAM_RS_MIN)
        
    if defense_score >= 50:
        ra_pg = LEAGUE_RPG - ((defense_score - 50.0) / 50.0) * (LEAGUE_RPG - TEAM_RA_MIN)
    else:
        ra_pg = LEAGUE_RPG + ((50.0 - defense_score) / 50.0) * (TEAM_RA_MAX - LEAGUE_RPG)
        
    win_rate = (rs_pg ** PYTHAG_EXP) / ((rs_pg ** PYTHAG_EXP) + (ra_pg ** PYTHAG_EXP))
    return total_score, rs_pg, ra_pg, win_rate

def calc_log5(p_a, p_b):
    p_a = float(np.clip(p_a, 0.01, 0.99))
    p_b = float(np.clip(p_b, 0.01, 0.99))
    denom = p_a + p_b - 2 * p_a * p_b
    if denom == 0:
        return 0.5
    return (p_a - p_a * p_b) / denom

# ==========================================
# 3. 10개 구단 풀 리그 & 포스트시즌 시뮬레이션 엔진
# ==========================================
def get_default_league_df(my_team_name, s, b, bat, d, base):
    teams = [
        {"구단명": my_team_name, "선발": s, "타선": bat, "불펜": b, "수비": d, "주루": base},
        {"구단명": "서울 트윈스", "선발": 72, "타선": 68, "불펜": 66, "수비": 65, "주루": 64},
        {"구단명": "광주 타이거즈", "선발": 68, "타선": 75, "불펜": 62, "수비": 58, "주루": 65},
        {"구단명": "대구 라이온즈", "선발": 64, "타선": 70, "불펜": 60, "수비": 62, "주루": 50},
        {"구단명": "수원 위즈", "선발": 69, "타선": 58, "불펜": 63, "수비": 60, "주루": 48},
        {"구단명": "인천 랜더스", "선발": 52, "타선": 66, "불펜": 55, "수비": 54, "주루": 55},
        {"구단명": "잠실 베어스", "선발": 56, "타선": 52, "불펜": 58, "수비": 62, "주루": 58},
        {"구단명": "부산 자이언츠", "선발": 48, "타선": 60, "불펜": 45, "수비": 46, "주루": 52},
        {"구단명": "대전 이글스", "선발": 54, "타선": 46, "불펜": 48, "수비": 45, "주루": 44},
        {"구단명": "창원 다이노스", "선발": 42, "타선": 50, "불펜": 40, "수비": 44, "주루": 48},
    ]
    return pd.DataFrame(teams)

def make_unique_names(names):
    seen = {}
    unique_list = []
    for name in names:
        if name in seen:
            seen[name] += 1
            unique_list.append(f"{name}({seen[name]})")
        else:
            seen[name] = 1
            unique_list.append(name)
    return unique_list

def simulate_full_league(teams_df, total_games):
    rng = np.random.default_rng()
    n_teams = len(teams_df)
    games_per_pair = max(1, round(total_games / (n_teams - 1)))
    actual_games = games_per_pair * (n_teams - 1)
    
    records = []
    for idx, row in teams_df.iterrows():
        tot, rs_pg, ra_pg, wp = calc_team_metrics(
            row["선발"], row["불펜"], row["타선"], row["수비"], row["주루"]
        )
        records.append({
            "team_id": idx,
            "구단명": str(row["구단명"]),
            "종합점수": round(tot, 1),
            "예상R/G": round(rs_pg, 2),
            "예상RA/G": round(ra_pg, 2),
            "기대승률": wp,
            "경기": actual_games,
            "승": 0,
            "패": 0,
        })
        
    # 1:1 상대 승수 기록용 배열
    h2h_wins = np.zeros((n_teams, n_teams), dtype=int)
    
    # 모든 구단 쌍(i < j) 간 맞대결 시뮬레이션 (제로섬)
    for i in range(n_teams):
        for j in range(i + 1, n_teams):
            prob_i = calc_log5(records[i]["기대승률"], records[j]["기대승률"])
            game_probs = np.clip(rng.normal(loc=prob_i, scale=0.04, size=games_per_pair), 0.05, 0.95)
            wins_i = int(np.sum(rng.binomial(1, game_probs)))
            wins_j = games_per_pair - wins_i
            
            records[i]["승"] += wins_i
            records[i]["패"] += wins_j
            records[j]["승"] += wins_j
            records[j]["패"] += wins_i
            
            h2h_wins[i, j] = wins_i
            h2h_wins[j, i] = wins_j
            
    standings_df = pd.DataFrame(records)
    standings_df["승률"] = standings_df["승"] / standings_df["경기"]
    standings_df["득실마진"] = standings_df["예상R/G"] - standings_df["예상RA/G"]
    
    # 순위 정렬 (승률 -> 득실마진 -> 기대승률)
    standings_df = standings_df.sort_values(
        by=["승률", "득실마진", "기대승률"], ascending=False
    ).reset_index(drop=True)
    
    standings_df.insert(0, "순위", np.arange(1, n_teams + 1))
    first_w = standings_df.loc[0, "승"]
    first_l = standings_df.loc[0, "패"]
    standings_df["게임차"] = ((first_w - standings_df["승"]) + (standings_df["패"] - first_l)) / 2.0
    
    # 최종 순위 순서대로 1:1 상대 전적 매트릭스 생성
    sorted_ids = standings_df["team_id"].tolist()
    sorted_names = make_unique_names(standings_df["구단명"].tolist())
    
    matrix_data = []
    for r_idx, id_row in enumerate(sorted_ids):
        row_dict = {}
        for c_idx, id_col in enumerate(sorted_ids):
            col_label = sorted_names[c_idx]
            if r_idx == c_idx:
                row_dict[col_label] = "-"
            else:
                w = h2h_wins[id_row, id_col]
                l = h2h_wins[id_col, id_row]
                row_dict[col_label] = f"{w}승 {l}패"
        matrix_data.append(row_dict)
        
    h2h_df = pd.DataFrame(matrix_data, index=[f"{i+1}위 {name}" for i, name in enumerate(sorted_names)])
    
    return standings_df, h2h_df, actual_games, games_per_pair

def simulate_playoff_series(high_seed, low_seed, wins_needed, high_initial_wins=0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    prob_high = calc_log5(high_seed["기대승률"], low_seed["기대승률"])
    
    w_high = high_initial_wins
    w_low = 0
    game_logs = []
    game_num = 1
    
    while w_high < wins_needed and w_low < wins_needed:
        high_wins_game = rng.random() < prob_high
        if high_wins_game:
            w_high += 1
            winner_name = high_seed["구단명"]
        else:
            w_low += 1
            winner_name = low_seed["구단명"]
        game_logs.append(f"{game_num}차전: **{winner_name}** 승 ({w_high}:{w_low})")
        game_num += 1
        
    series_winner = high_seed if w_high == wins_needed else low_seed
    series_loser = low_seed if w_high == wins_needed else high_seed
    score_text = f"{w_high}승 {w_low}패" if w_high == wins_needed else f"{w_low}승 {w_high}패"
    
    return series_winner, series_loser, score_text, prob_high, game_logs

def simulate_multi_seasons(teams_df, total_games, n_seasons=100):
    rng = np.random.default_rng()
    n_teams = len(teams_df)
    games_per_pair = max(1, round(total_games / (n_teams - 1)))
    actual_games = games_per_pair * (n_teams - 1)
    
    team_infos = []
    for idx, row in teams_df.iterrows():
        tot, rs_pg, ra_pg, wp = calc_team_metrics(
            row["선발"], row["불펜"], row["타선"], row["수비"], row["주루"]
        )
        team_infos.append({
            "team_id": idx,
            "구단명": str(row["구단명"]),
            "종합점수": round(tot, 1),
            "기대승률": wp,
            "득실마진": rs_pg - ra_pg,
        })
        
    # n_seasons x n_teams 승수 누적 행렬 (벡터화 시뮬레이션)
    season_wins = np.zeros((n_seasons, n_teams), dtype=int)
    
    for i in range(n_teams):
        for j in range(i + 1, n_teams):
            prob_i = calc_log5(team_infos[i]["기대승률"], team_infos[j]["기대승률"])
            game_probs = np.clip(
                rng.normal(loc=prob_i, scale=0.04, size=(n_seasons, games_per_pair)),
                0.05, 0.95
            )
            wins_i = np.sum(rng.binomial(1, game_probs), axis=1)
            wins_j = games_per_pair - wins_i
            
            season_wins[:, i] += wins_i
            season_wins[:, j] += wins_j
            
    reg_first_counts = np.zeros(n_teams, dtype=int)
    ps_counts = np.zeros(n_teams, dtype=int)
    ks_champ_counts = np.zeros(n_teams, dtype=int)
    
    margins = np.array([t["득실마진"] for t in team_infos])
    exp_wps = np.array([t["기대승률"] for t in team_infos])
    
    for s in range(n_seasons):
        w_arr = season_wins[s]
        # 동률 타이브레이커: 승수 -> 득실마진 -> 기대승률 -> 미세 난수
        tie_noise = rng.random(n_teams) * 1e-6
        order = np.lexsort((tie_noise, exp_wps, margins, w_arr))[::-1]
        
        reg_first_counts[order[0]] += 1
        ps_counts[order[:5]] += 1
        
        # 계단식 포스트시즌 진행
        t1 = team_infos[order[0]]
        t2 = team_infos[order[1]]
        t3 = team_infos[order[2]]
        t4 = team_infos[order[3]]
        t5 = team_infos[order[4]]
        
        wc_win, _, _, _, _ = simulate_playoff_series(t4, t5, wins_needed=2, high_initial_wins=1, rng=rng)
        spo_win, _, _, _, _ = simulate_playoff_series(t3, wc_win, wins_needed=3, rng=rng)
        po_win, _, _, _, _ = simulate_playoff_series(t2, spo_win, wins_needed=3, rng=rng)
        ks_win, _, _, _, _ = simulate_playoff_series(t1, po_win, wins_needed=4, rng=rng)
        
        ks_champ_counts[ks_win["team_id"]] += 1
        
    summary_rows = []
    for i, info in enumerate(team_infos):
        avg_w = float(np.mean(season_wins[:, i]))
        summary_rows.append({
            "구단명": info["구단명"],
            "종합점수": info["종합점수"],
            "기대승률": info["기대승률"],
            "평균승수": round(avg_w, 1),
            "평균승률": avg_w / actual_games,
            "정규1위확률(%)": round((reg_first_counts[i] / n_seasons) * 100, 1),
            "가을야구확률(%)": round((ps_counts[i] / n_seasons) * 100, 1),
            "한국시리즈우승확률(%)": round((ks_champ_counts[i] / n_seasons) * 100, 1),
        })
        
    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["한국시리즈우승확률(%)", "가을야구확률(%)", "평균승수"],
        ascending=False
    ).reset_index(drop=True)
    summary_df.insert(0, "우승기대순위", np.arange(1, n_teams + 1))
    
    return summary_df, actual_games

# ==========================================
# 4. UI 구성 시작
# ==========================================
st.title("⚾ 프로야구 구단 전력 분석 & 풀 리그 시뮬레이터")
st.markdown(
    f"리그 평균 **{LEAGUE_RPG} R/G** · 피타고리안 지수 **{PYTHAG_EXP}** · 가을야구 커트라인 **{POSTSEASON_CUTOFF:.3f}** 적용"
)

# 사이드바
with st.sidebar:
    st.header("🏟️ 내 구단 & 리그 설정")
    team_name = st.text_input("내 구단명 입력", value="드림즈 베이스볼 클럽")
    total_games = st.number_input("시즌 경기 수 (9의 배수 권장)", min_value=54, max_value=162, value=144, step=9)
    
    with st.expander("📌 적용된 리그 통계 상수"):
        st.write(f"- 리그 평균 R/G: **{LEAGUE_RPG}**")
        st.write(f"- 팀 R/G 범위: **{TEAM_RS_MIN} ~ {TEAM_RS_MAX}**")
        st.write(f"- 팀 RA/G 범위: **{TEAM_RA_MIN} ~ {TEAM_RA_MAX}**")
        st.write(f"- 피타고리안 지수: **{PYTHAG_EXP}**")
        st.write(f"- 홈 어드밴티지: **미적용 ({HOME_ADVANTAGE})**")

    st.divider()
    st.subheader("⚡ 내 구단 컬러 프리셋")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.button("👑 우승권 강팀", on_click=set_preset, args=(80, 75, 78, 72, 68), use_container_width=True)
        st.button("🔥 핵타선 군단", on_click=set_preset, args=(42, 40, 88, 45, 70), use_container_width=True)
    with col_p2:
        st.button("🛡️ 지키는 야구", on_click=set_preset, args=(78, 84, 42, 80, 50), use_container_width=True)
        st.button("🌱 리빌딩 약팀", on_click=set_preset, args=(30, 34, 35, 38, 42), use_container_width=True)
    st.button("⚖️ 리그 평균 (50점)", on_click=set_preset, args=(50, 50, 50, 50, 50), use_container_width=True)

# 상단: 내 구단 전력 입력 및 요약
left_col, right_col = st.columns([1, 1.35], gap="large")

with left_col:
    st.subheader(f"📊 [{team_name}] 5대 전력 설정")
    starter = st.slider("1. 선발 투수진 (가중치 30%)", 0, 100, key="starter")
    batting = st.slider("2. 타선 / 득점력 (가중치 25%)", 0, 100, key="batting")
    bullpen = st.slider("3. 불펜 투수진 (가중치 20%)", 0, 100, key="bullpen")
    defense = st.slider("4. 수비력 (가중치 15%)", 0, 100, key="defense")
    baserunning = st.slider("5. 주루 및 작전 (가중치 10%)", 0, 100, key="baserunning")

tot_score, rs_pg, ra_pg, win_rate = calc_team_metrics(starter, bullpen, batting, defense, baserunning)
exp_wins = round(total_games * win_rate)

with right_col:
    st.subheader("📈 내 구단 기대 성적 프리뷰")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("종합 평점", f"{tot_score:.1f} 점", f"{tot_score - 50:+.1f}")
    m2.metric("기대 승률", f"{win_rate:.3f}", f"{(win_rate - POSTSEASON_CUTOFF):+.3f} (5위 대비)")
    m3.metric("예상 승패", f"{exp_wins}승 {total_games - exp_wins}패", f"{total_games}경기 기준")
    m4.metric("예상 득실점", f"{rs_pg:.2f} / {ra_pg:.2f}", f"마진 {rs_pg - ra_pg:+.2f}")

    chart_df = pd.DataFrame({
        "전력 영역": ["선발(30%)", "타선(25%)", "불펜(20%)", "수비(15%)", "주루/작전(10%)"],
        "점수": [starter, batting, bullpen, defense, baserunning]
    }).set_index("전력 영역")
    st.bar_chart(chart_df, height=210)

st.divider()

# 10개 구단 공통 전력 편집기 (풀 리그 & 100시즌 분석 공용)
st.subheader("📋 10개 구단 리그 전력표 (클릭하여 구단명·점수 직접 수정)")
sync_my_team = st.checkbox("1번 구단에 위에서 설정한 내 구단 이름과 전력 점수 자동 반영하기", value=True)

if "league_df" not in st.session_state:
    st.session_state.league_df = get_default_league_df(
        team_name, starter, bullpen, batting, defense, baserunning
    )
    
if sync_my_team:
    st.session_state.league_df.loc[0, "구단명"] = team_name
    st.session_state.league_df.loc[0, "선발"] = starter
    st.session_state.league_df.loc[0, "타선"] = batting
    st.session_state.league_df.loc[0, "불펜"] = bullpen
    st.session_state.league_df.loc[0, "수비"] = defense
    st.session_state.league_df.loc[0, "주루"] = baserunning

col_reset, _ = st.columns([1, 3])
with col_reset:
    if st.button("🔄 10개 구단 기본 전력표로 초기화"):
        st.session_state.league_df = get_default_league_df(
            team_name, starter, bullpen, batting, defense, baserunning
        )
        st.rerun()

edited_df = st.data_editor(
    st.session_state.league_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "구단명": st.column_config.TextColumn("구단명", required=True),
        "선발": st.column_config.NumberColumn("선발(30%)", min_value=0, max_value=100, step=1, required=True),
        "타선": st.column_config.NumberColumn("타선(25%)", min_value=0, max_value=100, step=1, required=True),
        "불펜": st.column_config.NumberColumn("불펜(20%)", min_value=0, max_value=100, step=1, required=True),
        "수비": st.column_config.NumberColumn("수비(15%)", min_value=0, max_value=100, step=1, required=True),
        "주루": st.column_config.NumberColumn("주루(10%)", min_value=0, max_value=100, step=1, required=True),
    }
)
st.session_state.league_df = edited_df.fillna(50)

st.divider()

# 하단 탭 구성
tab_league, tab_monte, tab_single, tab_h2h = st.tabs([
    "🏆 단일 시즌 풀 리그 & 상대 전적표",
    "📊 100시즌 반복 시뮬레이션 (우승 확률)",
    "🎮 내 구단 단독 시즌 시뮬레이션",
    "⚔️ 1:1 맞대결 승률 계산기 (Log5)"
])

# ==========================================
# TAB 1: 단일 시즌 풀 리그, 1:1 상대 전적표 & 포스트시즌
# ==========================================
with tab_league:
    st.markdown("위 10개 구단 전력표를 바탕으로 **한 시즌 정규리그 순위표, 구단별 1:1 상대 전적 매트릭스, 계단식 포스트시즌**을 시뮬레이션합니다.")

    if st.button("🏟️ 단일 시즌 개막 및 포스트시즌 토너먼트 실행!", type="primary", use_container_width=True):
        standings_df, h2h_df, actual_games, games_per_pair = simulate_full_league(
            st.session_state.league_df, total_games
        )
        
        st.subheader(f"📋 정규시즌 최종 순위표 (팀당 {actual_games}경기 / 팀 간 {games_per_pair}차전)")
        
        display_df = standings_df.copy()
        display_df["가을야구"] = display_df["순위"].apply(lambda r: "🎫 진출" if r <= 5 else "-")
        display_df["승률"] = display_df["승률"].map("{:.3f}".format)
        display_df["기대승률"] = display_df["기대승률"].map("{:.3f}".format)
        display_df["게임차"] = display_df["게임차"].map("{:.1f}".format)
        display_df["예상R/G"] = display_df["예상R/G"].map("{:.2f}".format)
        display_df["예상RA/G"] = display_df["예상RA/G"].map("{:.2f}".format)
        
        st.dataframe(
            display_df[["순위", "가을야구", "구단명", "경기", "승", "패", "승률", "게임차", "종합점수", "기대승률", "예상R/G", "예상RA/G"]],
            use_container_width=True,
            hide_index=True
        )
        
        # 내 구단 순위 안내
        my_row = standings_df[standings_df["team_id"] == 0]
        if not my_row.empty:
            my_rank = int(my_row.iloc[0]["순위"])
            my_w = int(my_row.iloc[0]["승"])
            my_l = int(my_row.iloc[0]["패"])
            my_wp = float(my_row.iloc[0]["승률"])
            my_actual_name = str(my_row.iloc[0]["구단명"])
            if my_rank <= 5:
                st.success(f"🎉 **[{my_actual_name}]** 정규시즌 **{my_rank}위** ({my_w}승 {my_l}패, 승률 {my_wp:.3f})로 포스트시즌에 진출했습니다!")
            else:
                st.warning(f"🍂 **[{my_actual_name}]** 정규시즌 **{my_rank}위** ({my_w}승 {my_l}패, 승률 {my_wp:.3f})로 가을야구 진출에 실패했습니다.")

        # 1:1 상대 전적 매트릭스 표시
        st.divider()
        st.subheader(f"⚔️ 구단별 1:1 상대 전적표 (팀 간 {games_per_pair}차전 기준)")
        st.caption("가로줄(행) 구단이 세로줄(열) 구단을 상대로 기록한 승·패 전적입니다. (최종 순위순 정렬)")
        st.dataframe(h2h_df, use_container_width=True)

        # 계단식 포스트시즌 결과 표시
        st.divider()
        st.subheader("🔥 포스트시즌 계단식 토너먼트 결과")
        
        t1 = standings_df.iloc[0].to_dict()
        t2 = standings_df.iloc[1].to_dict()
        t3 = standings_df.iloc[2].to_dict()
        t4 = standings_df.iloc[3].to_dict()
        t5 = standings_df.iloc[4].to_dict()
        
        wc_win, wc_lose, wc_score, wc_prob, wc_logs = simulate_playoff_series(
            t4, t5, wins_needed=2, high_initial_wins=1
        )
        spo_win, spo_lose, spo_score, spo_prob, spo_logs = simulate_playoff_series(
            t3, wc_win, wins_needed=3
        )
        po_win, po_lose, po_score, po_prob, po_logs = simulate_playoff_series(
            t2, spo_win, wins_needed=3
        )
        ks_win, ks_lose, ks_score, ks_prob, ks_logs = simulate_playoff_series(
            t1, po_win, wins_needed=4
        )
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("#### 1️⃣ 와일드카드 결정전")
            st.caption(f"4위 {t4['구단명']} vs 5위 {t5['구단명']}")
            st.info(f"**승자: {wc_win['구단명']}** ({wc_score})")
            with st.expander("경기 상세 로그"):
                st.write(f"4위 1승 어드밴티지 적용 (단판 승률 {wc_prob*100:.1f}%)")
                for log in wc_logs:
                    st.markdown(f"- {log}")
                    
        with c2:
            st.markdown("#### 2️⃣ 준플레이오프 (5전 3선승)")
            st.caption(f"3위 {t3['구단명']} vs WC {wc_win['구단명']}")
            st.info(f"**승자: {spo_win['구단명']}** ({spo_score})")
            with st.expander("경기 상세 로그"):
                st.write(f"3위 단판 승률: {spo_prob*100:.1f}%")
                for log in spo_logs:
                    st.markdown(f"- {log}")
                    
        with c3:
            st.markdown("#### 3️⃣ 플레이오프 (5전 3선승)")
            st.caption(f"2위 {t2['구단명']} vs 준PO {spo_win['구단명']}")
            st.info(f"**승자: {po_win['구단명']}** ({po_score})")
            with st.expander("경기 상세 로그"):
                st.write(f"2위 단판 승률: {po_prob*100:.1f}%")
                for log in po_logs:
                    st.markdown(f"- {log}")
                    
        with c4:
            st.markdown("#### 4️⃣ 한국시리즈 (7전 4선승)")
            st.caption(f"1위 {t1['구단명']} vs PO {po_win['구단명']}")
            st.success(f"🏆 **우승: {ks_win['구단명']}** ({ks_score})")
            with st.expander("경기 상세 로그"):
                st.write(f"1위 단판 승률: {ks_prob*100:.1f}%")
                for log in ks_logs:
                    st.markdown(f"- {log}")

        st.balloons()
        st.success(f"👑 **최종 한국시리즈 챔피언: {ks_win['구단명']}** (정규시즌 {int(ks_win['순위'])}위 / 시리즈 전적 {ks_score})")

# ==========================================
# TAB 2: 100시즌 반복 시뮬레이션 (우승 확률 차트)
# ==========================================
with tab_monte:
    st.markdown("현재 설정된 10개 구단 전력으로 **정규시즌과 포스트시즌을 100번 반복 실행**해 각 구단의 우승 확률과 가을야구 진출 확률을 계산합니다.")
    
    sim_count = st.select_slider("반복 시뮬레이션 시즌 수 선택", options=[100, 300, 500, 1000], value=100)
    
    if st.button(f"📈 {sim_count}시즌 연속 시뮬레이션 및 우승 확률 분석 실행!", type="primary", use_container_width=True):
        summary_df, actual_games = simulate_multi_seasons(
            st.session_state.league_df, total_games, n_seasons=sim_count
        )
        
        top_champ = summary_df.iloc[0]
        st.success(
            f"🏆 **{sim_count}시즌 분석 결과 최다 우승 후보: [{top_champ['구단명']}]** "
            f"(한국시리즈 우승 확률 **{top_champ['한국시리즈우승확률(%)']:.1f}%** / 가을야구 진출 확률 **{top_champ['가을야구확률(%)']:.1f}%**)"
        )
        
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.markdown("#### 🏆 구단별 한국시리즈 우승 확률 (%)")
            ks_chart_df = summary_df[["구단명", "한국시리즈우승확률(%)"]].set_index("구단명")
            st.bar_chart(ks_chart_df, height=300)
            
        with col_chart2:
            st.markdown("#### 🎫 구단별 가을야구(5위 이내) 진출 확률 (%)")
            ps_chart_df = summary_df[["구단명", "가을야구확률(%)"]].set_index("구단명")
            st.bar_chart(ps_chart_df, height=300)
            
        st.markdown(f"#### 📋 {sim_count}시즌 시뮬레이션 종합 리포트 (팀당 {actual_games}경기 기준)")
        disp_summary = summary_df.copy()
        disp_summary["기대승률"] = disp_summary["기대승률"].map("{:.3f}".format)
        disp_summary["평균승률"] = disp_summary["평균승률"].map("{:.3f}".format)
        disp_summary["평균승수"] = disp_summary["평균승수"].map("{:.1f}승".format)
        disp_summary["정규1위확률(%)"] = disp_summary["정규1위확률(%)"].map("{:.1f}%".format)
        disp_summary["가을야구확률(%)"] = disp_summary["가을야구확률(%)"].map("{:.1f}%".format)
        disp_summary["한국시리즈우승확률(%)"] = disp_summary["한국시리즈우승확률(%)"].map("{:.1f}%".format)
        
        st.dataframe(disp_summary, use_container_width=True, hide_index=True)

# ==========================================
# TAB 3: 내 구단 단독 시즌 시뮬레이션
# ==========================================
with tab_single:
    st.markdown(f"기대 승률(**{win_rate:.3f}**)을 기준으로 내 구단의 한 시즌 승률 흐름만 빠르게 테스트합니다.")
    if st.button("🚀 단독 시즌 시뮬레이션 실행", type="primary"):
        rng = np.random.default_rng()
        game_probs = np.clip(rng.normal(loc=win_rate, scale=0.075, size=total_games), 0.05, 0.95)
        results = rng.binomial(1, game_probs)
        
        sim_wins = int(np.sum(results))
        sim_losses = total_games - sim_wins
        sim_win_rate = sim_wins / total_games
        
        cum_wins = np.cumsum(results)
        games_played = np.arange(1, total_games + 1)
        cum_win_rates = cum_wins / games_played
        
        if sim_win_rate >= POSTSEASON_CUTOFF:
            st.success(f"**결과: {sim_wins}승 {sim_losses}패 (승률 {sim_win_rate:.3f})** — 가을야구 진출 커트라인({POSTSEASON_CUTOFF:.3f}) 돌파!")
        else:
            st.warning(f"**결과: {sim_wins}승 {sim_losses}패 (승률 {sim_win_rate:.3f})** — 가을야구 커트라인({POSTSEASON_CUTOFF:.3f}) 미달")
            
        trend_df = pd.DataFrame({
            "경기 수": games_played,
            "누적 승률": cum_win_rates,
            "가을야구 기준선 (0.520)": [POSTSEASON_CUTOFF] * total_games
        }).set_index("경기 수")
        st.line_chart(trend_df, height=240)

# ==========================================
# TAB 4: 1:1 맞대결 승률 계산기
# ==========================================
with tab_h2h:
    opp_col1, opp_col2 = st.columns([1, 1])
    with opp_col1:
        opp_name = st.text_input("상대 구단명", value="라이벌 구단")
        opp_win_rate = st.slider("상대 팀 시즌 기대 승률", 0.300, 0.700, 0.520, step=0.005)
    with opp_col2:
        h2h_prob = calc_log5(win_rate, opp_win_rate)
        st.metric(
            label=f"[{team_name}] vs [{opp_name}] 단판 맞대결 승리 확률",
            value=f"{h2h_prob * 100:.1f} %",
            delta=f"상대 승리 확률 {(1 - h2h_prob) * 100:.1f}%"
        )
        st.progress(float(h2h_prob))
