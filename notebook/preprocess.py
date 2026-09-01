import pandas as pd
import numpy as np
from webcolors import names

class TitanicPreprocessor:

    def __init__(self):
        self.og_train_df = None
        self.og_test_df = None

        self.prcd_train_df = None
        self.prcd_test_df = None

        self.TITLE_ALIASES = {
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Mme": "Mrs",
        "Miss": "Miss",
        "Ms": "Miss",
        "Mlle": "Miss",
        "Master": "Master",
        }
        self.TITLE_ORDER = ["Master", "Miss", "Mr", "Mrs", "Other"]  
        self.MIN_GROUP = 5
        self.EMBARKED_KEYS = [["Pclass", "Sex", "Deck"], ["Pclass", "Deck"], ["Pclass", "Sex"], ["Pclass"]]
        self.MIN_VOTES = 10
        self.FARE_KEYS = [["Pclass", "Embarked", "Companion"], ["Pclass", "Embarked"], ["Pclass"]]
        self.MIN_FARE_SAMPLES = 20
        self.SKEW_TOLERANCE = 0.1  # (평균-중앙값)/중앙값이 이보다 크면 치우친 분포로 보고 중앙값을 쓴다


    def extract_title(self, names: pd.Series) -> pd.Series:
        """이름에서 호칭을 뽑아 Mr / Mrs / Miss / Master / Other 다섯 범주로 리턴합니다.

        타이타닉의 이름은 "Braund, Mr. Owen Harris"처럼 성과 이름 사이에 호칭이 들어갑니다.
        호칭 하나에 성별·혼인 여부·연령대가 함께 담겨 있어(Master는 소년, Miss는 미혼 여성)
        Age 대치 기준으로 쓸모가 큽니다. 수가 적은 직함·귀족 호칭(Dr, Rev, Col, Lady ...)은
        그룹당 표본이 몇 명뿐이라 통계가 흔들리므로 'Other'로 묶습니다.
        """
        raw = names.str.extract(r",\s*([^.]*)\.", expand=False).str.strip()
        return raw.map(lambda title: np.nan if pd.isna(title) else self.TITLE_ALIASES.get(title, "Other"))

    def extract_deck(self, cabin: pd.Series) -> pd.Series:
        """객실 번호에서 덱(맨 앞 알파벳)을 뽑아 리턴합니다. Cabin이 결측이면 Unknown으로 채웁니다."""
        return cabin.str.strip().str[0].fillna("Unknown")

    def extract_side(self, cabin: pd.Series) -> pd.Series:
        """객실 번호에서 객실 방향(Port / Starboard)을 뽑아 리턴합니다.

        타이타닉의 객실 번호는 짝수가 좌현(Port), 홀수가 우현(Starboard)입니다.
        "C23 C25 C27"처럼 여러 객실이 적힌 경우 첫 번째 객실 번호를 기준으로 삼고,
        "T"나 "F"처럼 숫자가 없으면 방향을 알 수 없으므로 Missing으로 둡니다.
        Cabin 자체가 결측이면 Unknown으로 둡니다. (Unknown과 구분해야 합니다.)
        """
        number = cabin.str.strip().str.split().str[0].str.extract(r"(\d+)", expand=False)
        # na_action="ignore"가 없으면 NaN도 함수에 들어가 홀수로 취급된다
        side = number.astype(float).map(lambda n: "Port" if n % 2 == 0 else "Starboard", na_action="ignore")
        # Cabin은 있는데 숫자가 없는 경우만 Missing (Cabin 결측은 Unknown으로 남긴다)
        return side.where(number.notna(), cabin.notna().map({True: "Missing", False: "Unknown"}))

    def recoverable_by_key(self, df: pd.DataFrame, key_col: str, value_col: str) -> pd.Series:
        """키를 공유하는 다른 행의 값으로 결측을 채울 수 있는지 집계해 복원값을 리턴합니다.

        같은 티켓 번호를 쓴 승객처럼 '같은 키를 공유하는 행'은 값이 같아야 하는 경우가 있습니다.
        이때 한 행이 비어 있어도 같은 키의 다른 행에 값이 있으면 그대로 옮겨 채울 수 있습니다.
        키 안에서 서로 다른 값이 둘 이상 관측되면 어느 값을 쓸지 정할 수 없으므로 '충돌'로
        따로 세고, 복원값에서도 제외합니다.

        Args:
            df: 대상 DataFrame. train/test를 합쳐서 넣어야 파일에 나뉘어 있는 키까지 묶입니다.
            key_col: 값을 공유하는 기준이 되는 키 컬럼(예: Ticket).
            value_col: 복원 대상 컬럼(예: Cabin).

        Returns:
            복원값. 복원값은 df와 같은 길이의 Series이고, 복원할 수 없는 행은 NaN입니다.
        """
        grouped = df.groupby(key_col)[value_col]
        n_values = grouped.transform("nunique")  # 키별로 관측된 서로 다른 값의 수
        first_value = grouped.transform("first")  # 키별 첫 관측값 (NaN은 건너뜀)

        missing = df[value_col].isna()
        recoverable = missing & (n_values == 1)

        return first_value.where(recoverable)

    def choose_age(self, title, pclass) -> float:
        """Title × Pclass -> Title -> 전체 중앙값 순으로 물러나며 대치값을 리턴합니다."""

        by_combo = self.prcd_train_df.groupby(["Title", "Pclass"])["Age"].agg(["median", "count"])
        by_title = self.prcd_train_df.groupby("Title")["Age"].agg(["median", "count"])
        overall_median = self.prcd_train_df["Age"].median()

        combo_median = by_combo.loc[by_combo["count"] >= self.MIN_GROUP, "median"].to_dict()
        title_median = by_title.loc[by_title["count"] >= self.MIN_GROUP, "median"].to_dict()
        
        if (title, pclass) in combo_median:
            return combo_median[(title, pclass)]
        if title in title_median:
            return title_median[title]
        return overall_median

    def choose_embarked(self, row: pd.Series, pool: pd.DataFrame) -> str:
        """조건을 넓혀 가며 표본이 충분한 첫 그룹의 최빈값을 리턴합니다."""
        known = pool[pool["Embarked"].notna()]
        for keys in self.EMBARKED_KEYS:
            # 기준 컬럼이 결측인 행(예: Deck 없음)은 어느 그룹에도 걸리지 않으므로 자동으로 다음 단계로 넘어간다
            group = known[np.logical_and.reduce([known[key].eq(row[key]) for key in keys])]
            if len(group) >= self.MIN_VOTES:
                votes = group["Embarked"].value_counts()
                return votes.index[0]
        votes = known["Embarked"].value_counts()
        return votes.index[0]

    def preprocess(self, original_train_location: str, original_test_location: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        파생변수 생성 및 결측값 전처리.
        대처 대상:
            - Name: Title 파생. Mr / Mrs / Miss / Master / Other 다섯 범주로 묶는다.
                - 파생 컬럼: Title
            - Age: Title × Pclass 조합의 중앙값으로 대치. 
                - 파생컬럼: hasAge
            - Cabin: 같은 티켓 번호를 소지한 다른 손님의 Cabin으로 대치. 
                - 파생 컬럼: hasCabin, Deck, Side 
            - Embarked: Embarked를 제외한 다른 컬럼이 같은 승객군 안에서의 최빈값으로 대치.
            - Fare: test_df 안에서 조건이 같은 승객들의 평균 또는 중앙값으로 대치 (분산·치우침을 보고 선택).
        파생 컬럼:
            - hasAge: Age 결측 여부 (0/1)
            - hasCabin: Cabin 결측 여부 (0/1)
            - Title: Name에서 뽑은 호칭
            - Deck: Cabin에서 뽑은 덱 (맨 앞 알파벳)
            - Side: Cabin에서 뽑은 객실 방향 (Port / Starboard / Missing / Unknown)
            - Companion: SibSp + Parch (Fare 대치용)
        Args:
            original_train_location: 원본 train.csv 경로
            original_test_location: 원본 test.csv 경로
        Returns:
            대치본 train_df, test_df
        """

        self.og_train_df = pd.read_csv(original_train_location)
        self.og_test_df = pd.read_csv(original_test_location)

        # 원본(train_df, test_df)은 손대지 않고 대치본을 따로 만든다. 
        self.prcd_train_df = self.og_train_df.copy()
        self.prcd_test_df = self.og_test_df.copy()

        # 1. 파생 변수, 결측 플래그 생성
        for prcd, origin in ((self.prcd_train_df, self.og_train_df), (self.prcd_test_df, self.og_test_df)):
            prcd["hasAge"] = origin["Age"].notna().astype(int)
            prcd["hasCabin"] = origin["Cabin"].notna().astype(int)
            prcd["Title"] = self.extract_title(origin["Name"])
            prcd["Deck"] = self.extract_deck(origin["Cabin"])
            prcd["Side"] = self.extract_side(origin["Cabin"])
            prcd["Companion"] = origin["SibSp"] + origin["Parch"]
            prcd["isAlone"] = (prcd["Companion"] == 0).astype(int)
            prcd["hasSibSp"] = (origin["SibSp"] > 0).astype(int)
            prcd["hasParch"] = (origin["Parch"] > 0).astype(int)

        # 2. Age 대치 + Cabin 파생
        for prcd, origin in ((self.prcd_train_df, self.og_train_df), (self.prcd_test_df, self.og_test_df)):
            chosen = prcd.apply(lambda row: self.choose_age(row["Title"], row["Pclass"]), axis=1)
            prcd["Age"] = origin["Age"].fillna(chosen)

        # 3. Embarked (train 2건): 다른 컬럼이 같은 승객군의 최빈값으로 채운다.
        #    조건을 좁힐수록 그 승객과 닮은 표본만 남지만 표가 줄어 최빈값이 흔들리므로,
        #    표본이 MIN_VOTES 이상인 가장 구체적인 조합까지만 내려간다.
        for prcd, origin in ((self.prcd_train_df, self.og_train_df), (self.prcd_test_df, self.og_test_df)):
            missing_idx = origin[origin["Embarked"].isna()].index
            for idx in missing_idx:
                row = prcd.loc[idx]
                value = self.choose_embarked(row, prcd) 
                prcd.loc[idx, "Embarked"] = value


        # 4. Fare (test 1건): test_df 안에서 조건이 같은 승객들의 값으로 채운다.
        #    Fare는 티켓 1장의 총액이라 동반 인원 수에 따라 몇 배로 뛴다. 그래서 동반자 수까지 맞춘다.
        for prcd, origin in ((self.prcd_train_df, self.og_train_df), (self.prcd_test_df, self.og_test_df)):
            missing_idx = origin[origin["Fare"].isna()].index
            if missing_idx.empty:
                continue

            pool = prcd

            for idx in missing_idx:
                target = pool.loc[idx] 

                for keys in self.FARE_KEYS:
                    values = pool[np.logical_and.reduce([pool[key].eq(target[key]) for key in keys])]["Fare"].dropna()
                    if len(values) >= self.MIN_FARE_SAMPLES:
                        break

                mean, median = values.mean(), values.median()
                skewed = abs(mean - median) / median > self.SKEW_TOLERANCE
                prcd.loc[idx, "Fare"] = median if skewed else mean
                
        return self.prcd_train_df, self.prcd_test_df
