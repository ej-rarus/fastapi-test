# FastAPI Test Project

간단한 FastAPI 웹 애플리케이션입니다.

## 기능

- **홈 엔드포인트**: `/` - 기본 인사 메시지
- **핑 엔드포인트**: `/ping` - 서버 상태 확인
- **예측 엔드포인트**: `/predict` - 간단한 건강 위험도 예측

## 설치 및 실행

### 1. 가상환경 활성화
```bash
source venv/bin/activate
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 서버 실행
```bash
uvicorn app.main:app --reload
```

서버가 실행되면 `http://localhost:8000`에서 애플리케이션에 접근할 수 있습니다.

## API 엔드포인트

### GET /
기본 홈페이지
```json
{
  "msg": "Hello, FastAPI on Mac!"
}
```

### GET /ping
서버 상태 확인
```json
{
  "msg": "pong"
}
```

### POST /predict
건강 위험도 예측
- **Parameters**:
  - `age` (int): 나이
  - `bmi` (float): BMI 지수
  - `glucose` (float): 혈당 수치

- **Response**:
```json
{
  "age": 30,
  "bmi": 25.5,
  "glucose": 110,
  "risk": 0
}
```

## API 문서

서버 실행 후 다음 URL에서 자동 생성된 API 문서를 확인할 수 있습니다:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 기술 스택

- **FastAPI**: 웹 프레임워크
- **Uvicorn**: ASGI 서버
- **Python 3.13**: 프로그래밍 언어
