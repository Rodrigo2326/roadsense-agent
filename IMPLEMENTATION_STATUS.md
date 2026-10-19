# RoadSense — Implementation Status

Convención: `DONE` = ejecutado y verificado. `NOT VERIFIED` = escrito pero sin ejecutar.
`BLOCKED` = requiere una decisión o un recurso que no tengo.

Última actualización: 2026-10-16 (sesión 3).

---

## Fase 0 — Inspección y smoke test — **DONE**

| Ítem | Estado | Evidencia |
|---|---|---|
| Inspección de repo y toolchain | DONE | Repo vacío salvo el brief. Python 3.11.9 + 3.14.0, aws-cli 2.34.24, git 2.53. |
| Contrato real de Decider 2B | DONE | Extraído del **código fuente** del wheel `strands-decider 0.1.0`, no de blogs. Ver `docs/architecture.md`. |
| Smoke test Decider 2B | DONE | 3 decisiones reales en EC2. Salida abajo. |
| Smoke test Bedrock | DONE | `converse` real contra `us.anthropic.claude-haiku-4-5-20251001-v1:0`. |
| Smoke test Rekognition | DONE | `DetectLabels` + `DetectText` reales sobre imágenes sintéticas. |
| Infra AWS aprovisionada | DONE | Ver "Recursos AWS creados". |

### Smoke test Decider 2B — salida real (EC2, CPU)

Comando: `/home/ubuntu/venv/bin/python -u /home/ubuntu/smoke_decider.py`

```
checkpoint=StrandsAgents/strands-decider-2B-qwen3.5-v1-2610  device=cpu
carga del modelo: 39.8s   cache HF: 4.5 GB

[step1_no_evidence]        choice='detect_objects'          conf=0.350  3195ms
  probs: detect_objects 0.513 | read_sign_text 0.471 | request_better_evidence 0.009 | answer 0.007
  evidence_sufficient P(true)=0.203

[step2_after_detection]    choice='read_sign_text'          conf=0.974  3368ms
  probs: detect_objects 0.011 | read_sign_text 0.981 | request_better_evidence 0.003 | answer 0.005
  evidence_sufficient P(true)=0.307

[step3_blurry_ocr_failed]  choice='request_better_evidence' conf=0.823  3257ms
  probs: detect_objects 0.006 | read_sign_text 0.111 | request_better_evidence 0.867 | answer 0.016
  evidence_sufficient P(true)=0.246
```

Esto cubre tres criterios de aceptación del brief (§9) de forma medible:
- la decisión **cambió** según la evidencia acumulada (paso 1 → paso 2);
- la segunda decisión **depende** de lo que devolvió la primera herramienta;
- con evidencia pobre el modelo eligió **pedir mejor evidencia**, no inventar texto.

### Smoke test AWS — salida real (local)

Comando: `python scripts/smoke_aws.py`

```
synthetic_sign_clear.jpg   detect_objects ok quality=0.999 (1297ms)  Road, Sign, Symbol, Bus Stop, Road Sign...
                           read_sign_text ok quality=1.000 (556ms)   'STOP | AHEAD'
synthetic_sign_blurry.jpg  detect_objects ok quality=0.999 (297ms)
                           read_sign_text ok quality=0.984 (678ms)   'STOP | AHEAD'
bedrock haiku-4.5          BEDROCK_OK  usage={'inputTokens':16,'outputTokens':8}
```

---

## Recursos AWS creados (cuenta 521106215898, us-east-1)

Todos etiquetados `Project=roadsense`. Nada preexistente fue modificado.

| Recurso | Identificador | Coste |
|---|---|---|
| S3 bucket | `roadsense-demo-521106215898` (acceso público bloqueado) | centavos |
| IAM role | `roadsense-ec2-role` (+ `roadsense-ec2-profile`) | $0 |
| Key pair | `roadsense-key` → `~/.ssh/roadsense-key.pem` | $0 |
| Security group | `sg-010c4ec7103ba5dee` (SSH solo desde 38.250.154.87/32) | $0 |
| EC2 | `i-0c4ac9fece9f5625b` · c7i.2xlarge · 40 GB gp3 | **STOPPED** desde 2026-10-16. ~$0.36/h solo mientras esté running |
| EBS | `vol-04680bd27ecfc81b6` · 40 GB gp3 | ~$3.20/mes, se cobra aunque la instancia esté detenida |

El rol de instancia da permisos mínimos: `bedrock:InvokeModel`/`Converse`,
`rekognition:DetectLabels`/`DetectText`, y S3 solo sobre el bucket del proyecto.
**No hay credenciales de usuario en el servidor.**

---

## Hallazgos que cambian el diseño

1. **Decider 2B no está en Bedrock.** Es un adapter LoRA de HuggingFace (202 MB) sobre
   `Qwen/Qwen3.5-2B-Base` (4.56 GB) que corre localmente vía `pip install strands-decider`.
   Por eso existe el EC2.
2. **Latencia real ≈ 3.2 s por decisión en CPU**, con 2 preguntas por llamada. Es el
   coste dominante del ciclo; el benchmark debe reportarlo, no estimarlo.
   `transformers` avisa que `causal_conv1d` y `flash-linear-attention` no están
   instalados y usa la implementación de referencia — hay margen de optimización.
3. **`load_engine` usa `device="cuda"` por defecto.** Hay que pasar `device="cpu"`
   explícitamente o falla.
4. **La confianza de Decider es real y calibrada** (`(N·p_max−1)/(N−1)`), así que no
   hay que simular ninguna confianza en ningún punto del sistema.
5. **Las imágenes sintéticas "borrosas" no son suficientemente ilegibles**: con blur
   gaussiano 7 px Rekognition aún lee `STOP | AHEAD` con 0.984. Para el caso difícil
   del benchmark hace falta degradación más agresiva (blur + reducción de escala).
   Pendiente de calibrar en Fase 4.

---

## Fases 1 y 2 — **DONE**

Módulos: `decision_schema.py`, `state.py`, `policy.py`, `decider_client.py`,
`bedrock_client.py`, `telemetry.py`, `agent.py`, `cli.py`, `tools/{detect_objects,
read_sign_text,request_better_evidence}.py`.

**Tests: 39 pasan**, local y en EC2 (`python -m pytest tests/ -q`). Los tests usan
dobles deterministas; las integraciones reales se verifican con los smoke tests y con
las corridas end-to-end de abajo.

### End-to-end real (EC2, Decider 2B + Rekognition + Bedrock)

Caso nítido, pregunta que necesita las dos herramientas:

```
step 1: detect_objects   conf=0.776  P(suf)=0.196  3325ms
step 2: read_sign_text   conf=0.952  P(suf)=0.273  3532ms
step 3: answer           conf=0.501  P(suf)=0.572  3719ms
evidencia: Road/Sign/Symbol/Bus Stop (1.00) ; OCR 'STOP | AHEAD' (1.00)
respuesta: "No vehicles appear in the image. The sign reads "STOP | AHEAD"..."
```

La segunda decisión cambió por la evidencia de la primera, y `P(evidencia suficiente)`
subió 0.196 → 0.273 → 0.572 conforme se acumuló evidencia.

### Hallazgo medido: Decider repite herramientas ya ejecutadas

Con las 4 acciones siempre ofrecidas, Decider eligió acciones ilegales de forma
repetida y la política tuvo que vetarlas:

| Caso | Sin enmascarar | Con enmascarar |
|---|---|---|
| Sin imagen | eligió `detect_objects` (p=0.907) → vetado; 1 decisión desperdiciada | eligió `request_better_evidence` directo, **conf 0.962**, 0 vetos |
| Imagen ilegible | 2 vetos (insistió en `read_sign_text` a p=0.891 tras devolver vacío) | 0 vetos, ruta limpia |

Por eso `agent.run(mask_illegal_actions=True)` es el valor por defecto: solo se le
ofrecen a Decider las acciones legales en ese estado. El flag existe para poder medir
ambos brazos en el benchmark. Es un resultado honesto y reportable: **el guardrail
tipado hace trabajo real, y enmascarar sube la confianza del modelo en vez de pelearse
con él.**

### Caso de evidencia insuficiente — verificado

```
imagen ilegible    -> OCR no devuelve nada -> "No readable text could be extracted..."
                      (NO inventa texto)
baja confianza OCR -> OCR 'STOP | -' a 0.66 -> "The sign text is not confirmable...
                      low confidence (0.66)... a higher-resolution image would be needed"
sin imagen         -> request_better_evidence conf 0.962 -> outcome insufficient_evidence
```

### Calibración empírica de las imágenes difíciles

Rekognition DetectText resultó mucho más robusto de lo esperado. Barrido real sobre
la misma escena, reduciendo el ancho efectivo:

| ancho efectivo | OCR |
|---|---|
| 110 px | lee `STOP \| AHEAD` a 0.991 |
| 80 px | lee `STOP \| -` a **0.658** → caso "no confirmable" |
| 60 px | lee `STOP \| I` a 0.827 |
| 45 px o menos | **no devuelve texto** → caso "ilegible" |

Los assets `synthetic_sign_lowconf.jpg` (80 px) y `synthetic_sign_illegible.jpg`
(45 px) salen de esta medición, no de una suposición.

### Coste: qué se reporta y qué no

- Rekognition: **$0.001/imagen**, verificado contra la AWS Pricing API (us-east-1,
  grupo "Image Async API", primer tramo).
- Bedrock: **coste omitido a propósito**. La Pricing API no publica tarifa para
  `claude-haiku-4-5` y la página pública de precios no la lista, así que inventarla
  sería falsear el dato. Los tokens sí son exactos. `pricing.json` permite añadir la
  tarifa si se consigue de la factura.
- Decider 2B: no se factura por token, corre en cómputo propio. Se reporta la hora de
  EC2 por separado.

---

## Fase 3 — Patrón reutilizable — **DONE**

`src/roadsense/pattern/evidence_seeking.py`: el bucle completo sin dominio
(`EvidenceSeeker`, `Action`, `Progress`, `Episode`). **`agent.py` fue refactorizado
para usarlo** — un patrón que el agente principal no usara sería decorativo.

- Ejemplo independiente sin imágenes: `examples/triage_example.py` (triaje de tickets
  de soporte). Demuestra la dependencia secuencial: `check_refund_window` no se ofrece
  en el paso 1 porque depende de una fecha de entrega que aún no se consultó, y
  aparece en el paso 2.
- Documentación: `docs/decision-pattern.md`.
- Tests del patrón: `tests/test_pattern.py` (13 tests, sin AWS ni imágenes).

El refactor rompió un test, que detectó una regresión real: el estado dejó de saber
cuántos pasos llevaba durante el bucle. Se arregló añadiendo `Progress` al contrato
de `render_state` — el presupuesto es parte de la decisión, no contabilidad alrededor.

## Fase 4 — Benchmark — **DONE**

**37 casos** (`eval/cases.jsonl`): 9 easy, 11 medium, 12 hard, 5 invalid-input;
19 respondibles y 18 donde lo correcto es decir que no alcanza. Cuatro brazos por el
**mismo bucle, mismas herramientas, mismo redactor** — solo cambia quién elige:
`keyword`, `exhaustive`, `llm` (Bedrock), `decider` (Strands Decider 2B).

Análisis completo en **`docs/findings.md`**. Resultados crudos en
`eval/results/report_full.md` y `report_nomask.md`, generados por la corrida.

### Resultado honesto: Decider no gana en todo

| | keyword | exhaustive | llm | decider |
|---|---|---|---|---|
| Tool set exacto | **100%** | 45.9% | 89.2% | 75.7% |
| Llamadas innecesarias | **0** | 20 | 4 | 9 |
| Conducta correcta | 89.2% | 89.2% | **100%** | 97.3% |
| Invalid-input correctos | 2/5 | 2/5 | **5/5** | **5/5** |
| Latencia selector mediana | ~0 ms | ~0 ms | 1559 ms | 9119 ms |
| Tokens facturables del selector | 0 | 0 | 23311 | **0** |

Una regla de palabras clave gana en exactitud de herramienta. Parte de eso es un
sesgo del propio benchmark y está declarado: `expected_tools` se deriva del
**enunciado de la pregunta**, que es justo lo que lee la regla.

Lo que la regla no puede hacer es notar que el archivo no existe. En los 3 casos de
entrada inválida (`rs-034`, `rs-036`, `rs-037`) responde igual; Decider pide mejor
evidencia. Esa es la diferencia entre un enrutador y una decisión sobre evidencia.

### Enmascaramiento medido a escala

Mismos 37 casos, brazo decider, solo cambia el menú ofrecido:

| | enmascarado | sin enmascarar |
|---|---|---|
| **Vetos de política** | **0** | **40** |
| Conducta correcta | 97.3% | 97.3% |
| Latencia selector mediana | 9119 ms | 10834 ms |

40 de las primeras elecciones de Decider eran ilegales en el estado en que se
tomaron. El resultado final apenas cambia porque el fallback recorre su propio
ranking, pero el modelo gastó masa de probabilidad en 40 opciones imposibles.

### El caso que Decider falló

`rs-029`: señal degradada a 65 px, OCR devuelve la letra `I` con **0.854** de
confianza. Se esperaba `insufficient`; respondió. Difícil culpar al modelo: el estado
decía que OCR había funcionado con 0.85. Rekognition estaba *confiadamente
equivocado*, no apropiadamente incierto. Se dejó el caso y no se tocó el umbral.

### Bug propio encontrado por el benchmark

El check de fidelidad marcó `rs-001`. No era falso positivo: `read_sign_text`
construía el resumen legible desde las detecciones `LINE` sin filtrar, mientras los
items venían de `WORD` filtradas por confianza. Una línea puede pasar el umbral
aunque su palabra no. El resumen decía `'STOP | AHEAD | the'` y Bedrock repitió
fielmente un texto que no estaba en la evidencia estructurada. Corregido
reconstruyendo la lectura desde las palabras conservadas, agrupadas por `ParentId`.
Dos tests de regresión añadidos.

---

## Fases siguientes

| Fase | Estado |
|---|---|
| 1 — Flujo end-to-end | **DONE** |
| 2 — Secuencia de evidencia | **DONE** |
| 3 — Patrón reutilizable | **DONE** |
| 4 — Benchmark | **DONE** |
| 5 — Demo y publicación | PARCIAL — `README.md`, `demo/demo_script.md` y `demo/assets_manifest.md` escritos. **Artículo, video y publicación del repo requieren tu autorización explícita.** |

## Tests

**57 pasan**, local y en EC2: `python -m pytest tests/ -q`.

| Archivo | Cubre |
|---|---|
| `test_decision_schema.py` | contrato cerrado de acciones, legalidad, fallback |
| `test_tool_routing.py` | la decisión cambia qué herramienta corre; Decider no ve la imagen |
| `test_sequential_loop.py` | la 2ª decisión ve la salida de la 1ª; presupuesto |
| `test_abstention.py` | evidencia insuficiente; regresión del bug de OCR |
| `test_invalid_inputs.py` | entradas inválidas, fallos de herramienta, respuestas malformadas |
| `test_pattern.py` | el patrón sin dominio, enmascarado vs veto |

## BLOCKED / pendiente de decisión

- Nada bloqueante técnicamente.
- **Requiere tu autorización:** publicar el repositorio, el artículo en Builder Center
  y el video. No se hizo ningún commit ni push.
- Pendiente opcional: fotografías reales con licencia compatible. Hoy el conjunto es
  100% sintético y generado por `scripts/make_assets.py` (licencia inequívoca, sin
  caras ni matrículas), pero no son fotos.


---

## Cómo retomar mañana

La instancia está **detenida**. El disco conserva el venv, el stack de torch y los
4.5 GB de pesos del modelo ya descargados: al arrancar no hay que reinstalar nada.

```bash
# 1. Arrancar y obtener la IP nueva (cambia en cada arranque: no hay Elastic IP)
aws ec2 start-instances --instance-ids i-0c4ac9fece9f5625b --region us-east-1
aws ec2 wait instance-running --instance-ids i-0c4ac9fece9f5625b --region us-east-1
IP=$(aws ec2 describe-instances --instance-ids i-0c4ac9fece9f5625b --region us-east-1      --query "Reservations[0].Instances[0].PublicIpAddress" --output text); echo $IP

# 2. Si tu IP pública cambió, reabrir SSH para la nueva
MYIP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --group-id sg-010c4ec7103ba5dee   --protocol tcp --port 22 --cidr $MYIP/32 --region us-east-1

# 3. Conectar
ssh -i ~/.ssh/roadsense-key.pem ubuntu@$IP

# 4. Re-verificar que Decider sigue vivo (~45 s, sin descargas)
/home/ubuntu/venv/bin/python -u /home/ubuntu/smoke_decider.py
```

Al terminar la sesión:

```bash
aws ec2 stop-instances --instance-ids i-0c4ac9fece9f5625b --region us-east-1
```

Local, sin EC2, funciona todo lo que no usa Decider:

```bash
C:/Users/rodri/.venvs/roadsense/Scripts/python.exe scripts/smoke_aws.py
```
