# Guion del video — objetivo 2:40–2:55

Instrucciones en español. **Las frases entre comillas son lo que dices, en inglés.**

El criterio del jurado es si se ve una decisión real de Decider y si esa decisión
cambia lo que pasa después. La interfaz web hace las dos cosas visibles, así que es
la protagonista. El terminal aparece solo al final, para las mediciones.

---

## Antes de grabar

Define primero tus identificadores (no están en el repo a propósito):

```bash
GPU_INSTANCE=i-xxxxxxxxxxxx   # la instancia con GPU
CPU_INSTANCE=i-xxxxxxxxxxxx   # la instancia CPU
SG_ID=sg-xxxxxxxxxxxx         # el security group del proyecto

# 1. Arrancar la GPU
aws ec2 start-instances --instance-ids $GPU_INSTANCE --region us-east-1
aws ec2 wait instance-running --instance-ids $GPU_INSTANCE --region us-east-1
IP=$(aws ec2 describe-instances --instance-ids $GPU_INSTANCE --region us-east-1 \
     --query "Reservations[0].Instances[0].PublicIpAddress" --output text); echo $IP

# 2. Abrir el puerto 8080 para tu IP de hoy
MYIP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --group-id $SG_ID \
  --protocol tcp --port 8080 --cidr $MYIP/32 --region us-east-1

# 3. Levantar el servidor
ssh -i ~/.ssh/roadsense-key.pem ubuntu@$IP "setsid /home/ubuntu/start_web.sh > /home/ubuntu/web.log 2>&1 < /dev/null &"

# 4. Esperar a que diga "model ready" (~45 s la primera vez, ~11 s después)
ssh -i ~/.ssh/roadsense-key.pem ubuntu@$IP "grep 'model ready' /home/ubuntu/web.log"
```

Abre `http://$IP:8080` y **haz una corrida de prueba completa antes de grabar.** La
píldora de arriba a la derecha debe decir `ready · cuda`. Si dice `loading`, espera.

Checklist de pantalla:
- Navegador a 1280 px de ancho o más; el panel derecho necesita espacio.
- Zoom al 100 %. Oculta marcadores y extensiones.
- Ten una segunda pestaña con `eval/results/report_gpu.md` ya abierta.
- Ten una tercera con `src/roadsense/pattern/evidence_seeking.py`.

---

## 0:00–0:18 · Qué es

**Pantalla:** la página cargada, sin correr nada todavía. Las tres tarjetas de arriba
(`step 1` / `then` / `step 2`) explican el ciclo solas.

> "This is RoadSense. It answers questions about road photographs — but it does not
> guess from one look. It has four things it can do, and it picks one at a time,
> seeing what came back before choosing the next. A two-billion-parameter model
> called **Strands Decider 2B** makes every one of those choices."

Señala la píldora `ready · cuda · strands-decider-2B-qwen3.5-v1-2610`.

## 0:18–1:00 · La decisión secuencial

**Haz:** clic en la muestra **"readable sign"**. Escribe o elige el chip
*"What vehicles appear and what text can be read on the sign?"*. Clic en **Run the agent**.

Deja que se dibuje solo. No hables encima del paso 1.

Cuando termine, señala en este orden:

> "Step one. Four actions on the menu, and Decider scores all four. It picks the
> detector — but look at the confidence: 0.77, with 0.83 on the detector and 0.16 on
> OCR. It is telling me both openings are reasonable, because they are."

> "Then the tool reports back. And step two is 0.98 on reading the sign. **The decision
> changed because the evidence changed.** That is the whole idea."

> "And the menu is shorter. The detector already ran, so it is withdrawn — Decider is
> never offered an action it is not allowed to take."

> "A hundred and sixty milliseconds a decision. On a GPU, this model is eight times
> faster than asking a large language model to do the same job — with zero billable
> tokens."

## 1:00–1:40 · Cuando la evidencia no alcanza

**Haz:** clic en la muestra **"degraded past reading"**. Pregunta:
*"What text can be read on the sign?"*. **Run the agent**.

> "Same sign, degraded until it cannot be read. OCR runs and comes back with nothing."

Señala el bloque de evidencia naranja.

> "It does not invent the word. The menu is down to two, and it answers — 0.862 against
> 0.138 for asking for a better photograph. But read the line underneath: confidence
> 0.724, and P(evidence enough) 0.270. It answers, and tells you the ground is thin."

Lee en voz alta el principio de la respuesta final:

> "'No readable text could be extracted.' Amazon Bedrock wrote that, and Bedrock never
> sees the image — only what the tools reported. It cannot describe what was not found."

## 1:40–2:00 · El guardrail, visible

**Haz:** clic en **remove image**. Deja la misma pregunta. **Run the agent**.

> "No image at all. And now the menu starts with only two options — both vision tools
> were withdrawn before the model ever saw them. It asks for a photograph, at 0.94
> confidence."

> "That is the reusable idea: **do not ask a model to respect a rule you can enforce by
> not offering the option.** Measured over thirty-seven cases, masking removed all
> forty illegal picks — and the model's confidence went up."

## 2:00–2:25 · El patrón, sin imágenes

**Pantalla:** pestaña con `evidence_seeking.py`. Desplázate hasta la construcción de
`EvidenceSeeker`.

> "The loop itself has no road, no image and no AWS call in it. You give it an action
> set, a way to render your state as text, and a legality function."

**Haz:** en una terminal, `python examples/triage_example.py`.

> "Here it triages a support ticket. Watch the menu: the refund lookup is not offered
> on step one, because it depends on a delivery date nobody has looked up yet. It
> appears on step two. Same engine, no image anywhere."

## 2:25–2:50 · Las mediciones, incluida la derrota

**Pantalla:** `eval/results/report_gpu.md`.

> "Thirty-seven cases, four selectors, same loop. And Decider does not win every row.
> A ten-line keyword rule picks the right tool more often than it does — partly
> because the ground truth comes from the question's wording, which is exactly what a
> keyword rule reads."

> "But the keyword rule cannot notice that the file is missing. On the invalid inputs
> it answers anyway, two out of five. Decider gets five out of five."

> "Seventy-eight tests, Apache-2.0, and every number came from a run, not from a
> spreadsheet."

## 2:50–2:55 · Cierre

> "Evidence-Seeking Tool Selection. The pattern is in the repo."

---

## Notas de grabación

- **No digas** "safe", "safety", "warns the driver", "detects hazards" ni nada que
  suene a seguridad vial. El proyecto es una demo de evidencia visual, no un sistema
  de seguridad, y decirlo mal invalida el encuadre que sostiene toda la documentación.
- Di **"Strands Decider 2B"** completo al menos dos veces. La submission se indexa así.
- No cortes las pausas de ~160 ms: son tan rápidas que el corte se nota más que la
  espera. Lo que sí debes cortar es el arranque del servidor.
- Si Bedrock devuelve un error en vivo, el agente cae a otro modelo solo y la traza lo
  registra. No lo escondas: si pasa, dilo — "Bedrock had no capacity for the first
  model, so it fell back, and the trace records which model answered."
- Graba cada bloque por separado. Es más fácil re-grabar 20 segundos que 3 minutos.

## Después de grabar

```bash
aws ec2 stop-instances --instance-ids $GPU_INSTANCE --region us-east-1
aws ec2 stop-instances --instance-ids $CPU_INSTANCE --region us-east-1
```
