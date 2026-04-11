@echo off
setlocal

cd /d "%~dp0"

echo Running remaining RAGAS evaluations...
echo Project: %CD%
echo.

python data/evaluation/ragas_evaluate.py --answers-path data/evaluation/results/answers_exaone-4.0-32b.json --model-name exaone-4.0-32b
if errorlevel 1 echo [WARN] exaone-4.0-32b failed with exit code %ERRORLEVEL%

python data/evaluation/ragas_evaluate.py --answers-path data/evaluation/results/answers_qwen3.5-35b-a3b.json --model-name qwen3.5-35b-a3b
if errorlevel 1 echo [WARN] qwen3.5-35b-a3b failed with exit code %ERRORLEVEL%

python data/evaluation/ragas_evaluate.py --answers-path data/evaluation/results/answers_qwen3.5-9b.json --model-name qwen3.5-9b
if errorlevel 1 echo [WARN] qwen3.5-9b failed with exit code %ERRORLEVEL%

python data/evaluation/ragas_evaluate.py --answers-path data/evaluation/results/answers_cohere.json --model-name cohere
if errorlevel 1 echo [WARN] cohere failed with exit code %ERRORLEVEL%

python data/evaluation/ragas_evaluate.py --answers-path data/evaluation/results/answers_openai.json --model-name openai
if errorlevel 1 echo [WARN] openai failed with exit code %ERRORLEVEL%

python data/evaluation/ragas_evaluate.py --answers-path data/evaluation/results/answers_anthropic.json --model-name anthropic
if errorlevel 1 echo [WARN] anthropic failed with exit code %ERRORLEVEL%

echo.
echo Done. Check data/evaluation/results/ragas_results.csv

endlocal
