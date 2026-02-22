try:
    from kiwipiepy import Kiwi
    _KIWI = Kiwi()

    def kiwi_tokenize(txt):
        """텍스트를 Kiwi로 토크나이즈"""
        try:
            return _KIWI.tokenize(txt, normalize=True)
        except TypeError:
            return _KIWI.tokenize(txt)

    def extract_nouns(text: str):
        """NN* (NNG, NNP, NNB 등) 명사만 추출"""
        toks = kiwi_tokenize(text)
        return [t.form for t in toks if t.tag.startswith("NN")]

except Exception as e:
    _KIWI = None

    def extract_nouns(text: str):
        """Kiwi 사용 불가 시 빈 리스트 반환"""
        print(f"[POS] Kiwi 사용 불가: {e}")
        return []
