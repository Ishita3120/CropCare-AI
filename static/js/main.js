(function () {
  "use strict";

  // ============================================================
  // ELEMENTS
  // ============================================================

  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const dropzoneEmpty = document.getElementById("dropzone-empty");
  const dropzonePreview = document.getElementById("dropzone-preview");
  const previewImg = document.getElementById("preview-img");
  const clearBtn = document.getElementById("clear-image");

  const form = document.getElementById("diagnosis-form");
  const analyzeBtn = document.getElementById("analyze-btn");
  const formError = document.getElementById("form-error");

  const report = document.getElementById("report");
  const reportInner = document.getElementById("report-inner");

  let selectedFile = null;

  let sensorTimer = null;

  // ============================================================
  // SIMULATED IOT SENSOR STREAM
  // ============================================================

  async function updateSensorData() {
    try {
      const res = await fetch("/sensor-data", {
        method: "GET",
        cache: "no-store"
      });

      if (!res.ok) {
        throw new Error("Sensor API returned " + res.status);
      }

      const sensor = await res.json();

      const temperatureInput =
        document.getElementById("temperature");

      const humidityInput =
        document.getElementById("humidity");

      const soilMoistureInput =
        document.getElementById("soil-moisture");
      temperatureInput.step = "any";
      humidityInput.step = "any";
      soilMoistureInput.step = "any";

      if (temperatureInput) {
        temperatureInput.value = sensor.temperature_c ?? "";
      }

      if (humidityInput) {
        humidityInput.value = sensor.humidity_pct ?? "";
      }

      if (soilMoistureInput) {
        soilMoistureInput.value =
          sensor.soil_moisture_pct ?? "";
      }

      console.log("IoT sensor reading:", sensor);

    } catch (err) {
      console.warn("IoT sensor update failed:", err);
    }
  }

  function startSensorStream() {
    // First reading immediately
    updateSensorData();

    // Next reading every 5 seconds
    sensorTimer = setInterval(
      updateSensorData,
      5000
    );
  }

  // Start simulated IoT stream
  startSensorStream();


  // ============================================================
  // DROPZONE
  // ============================================================

  dropzone.addEventListener("click", () => {
    fileInput.click();
  });


  dropzone.addEventListener("keydown", (e) => {

    if (e.key === "Enter" || e.key === " ") {

      e.preventDefault();
      fileInput.click();

    }

  });


  dropzone.setAttribute("tabindex", "0");
  dropzone.setAttribute("role", "button");
  dropzone.setAttribute(
    "aria-label",
    "Upload a leaf photo"
  );


  ["dragenter", "dragover"].forEach((evt) => {

    dropzone.addEventListener(evt, (e) => {

      e.preventDefault();
      dropzone.classList.add("dragover");

    });

  });


  ["dragleave", "drop"].forEach((evt) => {

    dropzone.addEventListener(evt, (e) => {

      e.preventDefault();
      dropzone.classList.remove("dragover");

    });

  });


  dropzone.addEventListener("drop", (e) => {

    const file = e.dataTransfer.files[0];

    if (file) {
      handleFile(file);
    }

  });


  fileInput.addEventListener("change", () => {

    if (fileInput.files[0]) {
      handleFile(fileInput.files[0]);
    }

  });


  clearBtn.addEventListener("click", (e) => {

    e.stopPropagation();
    resetFile();

  });


  // ============================================================
  // FILE HANDLING
  // ============================================================

  function handleFile(file) {

    const allowedTypes = [
      "image/png",
      "image/jpeg",
      "image/jpg",
      "image/webp"
    ];


    if (!allowedTypes.includes(file.type)) {

      showError(
        "That file type isn't supported. Use JPG, PNG or WEBP."
      );

      return;
    }


    if (file.size > 8 * 1024 * 1024) {

      showError(
        "Image is too large. Maximum size is 8 MB."
      );

      return;
    }


    clearError();

    selectedFile = file;

    const reader = new FileReader();


    reader.onload = (e) => {

      previewImg.src = e.target.result;

      dropzoneEmpty.hidden = true;
      dropzonePreview.hidden = false;

      analyzeBtn.disabled = false;

      // Hide previous result when a new image is selected
      report.hidden = true;

    };


    reader.readAsDataURL(file);

  }


  function resetFile() {

    selectedFile = null;

    fileInput.value = "";

    dropzoneEmpty.hidden = false;
    dropzonePreview.hidden = true;

    analyzeBtn.disabled = true;

    report.hidden = true;

    clearError();

  }


  // ============================================================
  // ERROR HANDLING
  // ============================================================

  function showError(message) {

    formError.textContent = message;
    formError.hidden = false;

  }


  function clearError() {

    formError.hidden = true;
    formError.textContent = "";

  }


  // ============================================================
  // FORM SUBMISSION
  // ============================================================

  form.addEventListener("submit", async (e) => {

    e.preventDefault();


    if (!selectedFile) {

      showError("Add a leaf photo first.");
      return;

    }


    clearError();


    const formData = new FormData();


    formData.append(
      "image",
      selectedFile
    );


    formData.append(
      "crop",
      document.getElementById("crop-select").value
    );


    const temp =
      document.getElementById("temperature").value;


    const hum =
      document.getElementById("humidity").value;


    const soil =
      document.getElementById("soil-moisture").value;


    if (temp) {

      formData.append(
        "temperature_c",
        temp
      );

    }


    if (hum) {

      formData.append(
        "humidity_pct",
        hum
      );

    }


    if (soil) {

      formData.append(
        "soil_moisture_pct",
        soil
      );

    }


    analyzeBtn.classList.add("loading");
    analyzeBtn.disabled = true;

    report.hidden = true;


    try {

      const res = await fetch(
        "/analyze",
        {
          method: "POST",
          body: formData
        }
      );


      const data = await res.json();


      // --------------------------------------------------------
      // BACKEND ERROR
      // --------------------------------------------------------


      if (!res.ok) {
        sessionStorage.setItem(
          "cropcareResult",
          JSON.stringify({
            ...data,
            isValidationError: true
          })
        );

        window.location.href = "/result";
        return;
      }


      sessionStorage.setItem(
        "cropcareResult",
        JSON.stringify(data)
      );

      window.location.href = "/result";
    }


    catch (err) {

      console.error(
        "Frontend analysis error:",
        err
      );


      showError(
        "The analysis could not be displayed. " +
        (err.message || err)
      );

    }


    finally {

      analyzeBtn.classList.remove("loading");

      analyzeBtn.disabled = false;

    }

  });


  // ============================================================
  // REPORT RENDERING
  // ============================================================

  function renderReport(data) {

    const sev = data.severity || {};
    const dis = data.disease || {};
    const dec = data.decision || {};
    const tr = data.treatment || null;
    const imgAnalysis = data.image_analysis || {};

    const env = data.env_used || {};

    const temperature =
      env.temperature_c !== null &&
        env.temperature_c !== undefined
        ? Number(env.temperature_c)
        : null;

    const humidity =
      env.humidity_pct !== null &&
        env.humidity_pct !== undefined
        ? Number(env.humidity_pct)
        : null;

    const soilMoisture =
      env.soil_moisture_pct !== null &&
        env.soil_moisture_pct !== undefined
        ? Number(env.soil_moisture_pct)
        : null;


    // ----------------------------------------------------------
    // DISEASE
    // ----------------------------------------------------------

    const diseaseLabel =
      dis.label !== undefined &&
        dis.label !== null
        ? String(dis.label).toLowerCase()
        : "unknown";


    const displayName =
      dis.display_name !== undefined &&
        dis.display_name !== null
        ? String(dis.display_name)
        : formatDiseaseName(diseaseLabel);


    const confidence =
      Number(dis.confidence || 0);


    // ----------------------------------------------------------
    // SEVERITY
    // ----------------------------------------------------------

    const severityClass =
      String(
        sev.severity_class || "low"
      ).toLowerCase();


    let affectedAreaPct;


    if (sev.affected_area_pct !== undefined) {

      affectedAreaPct =
        Number(sev.affected_area_pct);

    }

    else if (sev.score !== undefined) {

      affectedAreaPct =
        Number(sev.score) * 100;

    }

    else if (
      imgAnalysis.affected_area_ratio !== undefined
    ) {

      affectedAreaPct =
        Number(
          imgAnalysis.affected_area_ratio
        ) * 100;

    }

    else {

      affectedAreaPct = 0;

    }


    affectedAreaPct =
      Math.min(
        Math.max(affectedAreaPct, 0),
        100
      );


    // ----------------------------------------------------------
    // LEAF AREA
    // ----------------------------------------------------------

    let leafDetectedPct;


    if (sev.leaf_detected_pct !== undefined) {

      leafDetectedPct =
        Number(sev.leaf_detected_pct);

    }

    else if (
      imgAnalysis.leaf_area_ratio !== undefined
    ) {

      leafDetectedPct =
        Number(
          imgAnalysis.leaf_area_ratio
        ) * 100;

    }

    else {

      leafDetectedPct = 0;

    }


    leafDetectedPct =
      Math.min(
        Math.max(leafDetectedPct, 0),
        100
      );


    // ----------------------------------------------------------
    // IMAGE
    // ----------------------------------------------------------

    const imageUrl =
      data.image_url ||
      previewImg.src ||
      "";


    // ----------------------------------------------------------
    // HEALTH STATUS
    // ----------------------------------------------------------

    const isHealthy =
      diseaseLabel === "healthy";


    // ----------------------------------------------------------
    // FRIENDLY PERSONALITY
    // ----------------------------------------------------------

    let friendlyTitle;
    let friendlyMessage;
    let verdictLabel;


    if (isHealthy) {

      verdictLabel =
        "Plant feeling great 🌿";

      friendlyTitle =
        "I'm Healthy! 💚🌱";

      friendlyMessage =
        "Yay! I don't need any treatment right now. " +
        "Keep giving me sunshine, water and good care! ☀️💧";

    }

    else if (severityClass === "high") {

      verdictLabel =
        "Plant needs attention 🚨";

      friendlyTitle =
        "I Need Your Attention! 🌱🚨";

      friendlyMessage =
        "Uh-oh! Something is bothering me. " +
        "Let's take action before the problem gets worse.";

    }

    else if (severityClass === "medium") {

      verdictLabel =
        "Plant needs some care 🥺";

      friendlyTitle =
        "A Little Help, Please! 🌿🥺";

      friendlyMessage =
        "I've noticed some signs of disease. " +
        "Don't worry — let's help me get back to healthy! 💚";

    }

    else {

      verdictLabel =
        "Small signs detected 🌱";

      friendlyTitle =
        "I Could Use Some Care 🌱";

      friendlyMessage =
        "There's a small sign of trouble. " +
        "Let's keep an eye on me and follow the recommended guidance.";

    }


    // ----------------------------------------------------------
    // REPORT CSS
    // ----------------------------------------------------------

    reportInner.className =
      "report-inner severity-" +
      severityClass;


    // ----------------------------------------------------------
    // SEVERITY BAR
    // ----------------------------------------------------------

    const barWidthPct =
      affectedAreaPct;


    // ==========================================================
    // HTML
    // ==========================================================

    let html = "";


    // ----------------------------------------------------------
    // RESULT TOP
    // ----------------------------------------------------------

    html += `

      <div class="report-top">

        ${imageUrl
        ? `
              <img
                class="report-thumb"
                src="${imageUrl}"
                alt="Analyzed leaf photo"
              >
            `
        : ""
      }


        <div>

          <p class="report-verdict-label">

            ${verdictLabel}

            ·

            ${capitalize(data.crop || "Unknown")}

          </p>


          <h2 class="report-title">

            ${friendlyTitle}

          </h2>


          <p class="report-sub">

            ${friendlyMessage}

          </p>


          <span class="confidence-pill">

            ${Math.round(confidence * 100)}%
            AI confidence

          </span>

        </div>

      </div>

    `;


    // ----------------------------------------------------------
    // DETECTION DETAILS
    // ----------------------------------------------------------

    html += `

      <div
        class="detection-name"
        style="
          margin-top:18px;
          padding:12px 15px;
          border-radius:14px;
          background:#f3fbf6;
          color:#185c3b;
          font-size:13px;
          font-weight:700;
        "
      >

        🔎 Detected:
        ${displayName}

      </div>

    `;


    // ----------------------------------------------------------
    // METRICS
    // ----------------------------------------------------------

    html += `

      <div class="report-grid">

        <div>

          <p class="metric-label">
            🌿 Severity
          </p>


          <div class="severity-bar-track">

            <div
              class="severity-bar-fill ${severityClass}"
              style="width:${barWidthPct}%"
            ></div>

          </div>


          <p class="severity-value">

            ${capitalize(severityClass)}

            ·

            ${affectedAreaPct.toFixed(1)}%
            affected area

          </p>

        </div>


        <div>

          <p class="metric-label">
            🍃 Leaf detected
          </p>


          <div class="severity-bar-track">

            <div
              class="severity-bar-fill"
              style="
                width:${leafDetectedPct}%;
                background:#79c996
              "
            ></div>

          </div>


          <p class="severity-value">

            ${leafDetectedPct.toFixed(1)}%
            of the frame read as leaf material

          </p>

        </div>

      </div>

    `;
    // ----------------------------------------------------------
    // ENVIRONMENTAL CONDITIONS AT ANALYSIS
    // ----------------------------------------------------------

    html += `
  <div
    style="
      margin-top:18px;
      padding:18px;
      border-radius:18px;
      background:#f3fbf6;
      border:1px solid #d5eadc;
    "
  >
    <div
      style="
        font-size:13px;
        font-weight:800;
        color:#185c3b;
        margin-bottom:14px;
      "
    >
      📡 Environmental conditions at analysis
    </div>

    <div
      style="
        display:grid;
        grid-template-columns:repeat(3, 1fr);
        gap:12px;
      "
    >
      <div>
        <div style="font-size:12px;color:#668074;">
          🌡 Temperature
        </div>
        <strong style="font-size:16px;color:#123d2b;">
          ${temperature !== null
        ? temperature.toFixed(2) + " °C"
        : "Not available"
      }
        </strong>
      </div>

      <div>
        <div style="font-size:12px;color:#668074;">
          💧 Humidity
        </div>
        <strong style="font-size:16px;color:#123d2b;">
          ${humidity !== null
        ? humidity.toFixed(2) + " %"
        : "Not available"
      }
        </strong>
      </div>

      <div>
        <div style="font-size:12px;color:#668074;">
          🌱 Soil moisture
        </div>
        <strong style="font-size:16px;color:#123d2b;">
          ${soilMoisture !== null
        ? soilMoisture.toFixed(2) + " %"
        : "Not available"
      }
        </strong>
      </div>
    </div>

    <p
      style="
        margin:14px 0 0;
        font-size:11px;
        color:#668074;
      "
    >
      Reading captured when analysis was performed.
    </p>
  </div>
`;


    // ----------------------------------------------------------
    // TREATMENT GUIDANCE
    // ----------------------------------------------------------

    /*
      If a treatment exists but its dosage is 0–0,
      treat it as guidance available rather than
      chemical treatment.

      This is especially important for diseases
      where an expert-validated dosage has not
      been configured.
    */

    const hasTreatmentGuidance =
      tr &&
      Number(tr.min_dosage_ml_per_l || 0) <= 0 &&
      Number(tr.max_dosage_ml_per_l || 0) <= 0 &&
      diseaseLabel !== "healthy";


    // ----------------------------------------------------------
    // DECISION BANNER
    // ----------------------------------------------------------

    const bannerClass =
      dec.treat
        ? "treat"
        : hasTreatmentGuidance
          ? "guidance"
          : "no-treat";


    const bannerHeading =
      dec.treat
        ? "💊 I've got a treatment plan for you!"
        : hasTreatmentGuidance
          ? "🌱 I need some care — here's what to do"
          : "🥳 I'm healthy! No treatment needed";


    html += `

      <div
        class="decision-banner ${bannerClass}"
      >

        <p class="decision-heading">

          ${bannerHeading}

        </p>


        <p class="decision-text">

          ${isHealthy
        ? "Everything looks good! Keep taking care of me and check again later. 🌞🌿"
        : (dec.reason || friendlyMessage)
      }

        </p>


        ${dec.treat || hasTreatmentGuidance

        ? `

              <p
                class="decision-text"
                style="margin-top:8px"
              >

                ${dec.treat
          ? (
            dec.risk_note ||
            "Follow the recommended treatment guidance carefully."
          )
          : "Chemical spraying is disabled because no validated dosage is configured for this treatment."
        }

              </p>

            `

        : ""
      }

      </div>

    `;


    // ----------------------------------------------------------
    // TREATMENT
    // ----------------------------------------------------------

    if (
      (dec.treat || hasTreatmentGuidance) &&
      tr
    ) {

      html += `

        <div class="treatment-box">

          <h3>

            ${hasTreatmentGuidance
          ? "🌿 Treatment guidance"
          : "💊 Recommended treatment"
        }

          </h3>


          <dl class="treatment-facts">

            ${fact(
          "Product",
          tr.name || "Not specified"
        )}


            ${fact(
          "Active ingredient",
          tr.active_ingredient ||
          "Not specified"
        )}


            ${fact(
          "Dosage range",

          tr.min_dosage_ml_per_l !== undefined &&
            tr.max_dosage_ml_per_l !== undefined

            ? `${tr.min_dosage_ml_per_l}–${tr.max_dosage_ml_per_l} mL/L`

            : "Not specified"
        )}


            ${fact(
          "Suggested spray time",

          hasTreatmentGuidance

            ? "Not applicable — chemical spray disabled"

            : `${dec.spray_duration_s || 0}s (~${dec.spray_volume_ml || 0} mL)`
        )}


            ${fact(
          "Re-check in",

          `${dec.recheck_days || 7} day${(dec.recheck_days || 7) === 1
            ? ""
            : "s"
          }`
        )}

          </dl>


          ${tr.advice

          ? `

                <p
                  class="decision-text"
                  style="margin-top:14px"
                >

                  💡 ${tr.advice}

                </p>

              `

          : ""
        }

        </div>

      `;

    }


    else if (
      tr &&
      tr.advice
    ) {

      html += `

        <div class="treatment-box">

          <p class="decision-text">

            💡 ${tr.advice}

          </p>

        </div>

      `;

    }


    // ----------------------------------------------------------
    // HEALTHY CARE MESSAGE
    // ----------------------------------------------------------

    if (isHealthy) {

      html += `

        <div
          style="
            margin-top:18px;
            padding:18px;
            border-radius:18px;
            background:linear-gradient(
              135deg,
              #e8f7ee,
              #f5fbf7
            );
            border:1px solid #cfe9d8;
            text-align:center;
          "
        >

          <div
            style="
              font-size:32px;
              margin-bottom:6px;
            "
          >
            ☀️ 💧 🌱
          </div>


          <strong
            style="
              display:block;
              color:#145c3b;
              font-size:15px;
              margin-bottom:5px;
            "
          >
            Keep up the good care! 💚
          </strong>


          <span
            style="
              color:#668074;
              font-size:12px;
            "
          >
            Your plant is happy and doesn't need treatment right now.
          </span>

        </div>

      `;

    }


    // ----------------------------------------------------------
    // FOOTNOTE
    // ----------------------------------------------------------

    const method =
      dis.method ||
      "MobileNetV2 CNN + computer-vision severity analysis";


    html += `

      <p class="report-footnote">

        🤖 Analysis method:
        ${method}

        &mdash;

        This is an academic prototype
        and not a laboratory diagnosis.

      </p>

    `;


    // ----------------------------------------------------------
    // DISPLAY
    // ----------------------------------------------------------

    reportInner.innerHTML = html;

    report.hidden = false;


    report.scrollIntoView({
      behavior: "smooth",
      block: "start"
    });

  }


  // ============================================================
  // HELPERS
  // ============================================================

  function fact(label, value) {

    return `

      <div>

        <dt>${label}</dt>

        <dd>${value}</dd>

      </div>

    `;

  }


  function formatDiseaseName(label) {

    if (
      label === null ||
      label === undefined
    ) {

      return "Unknown";

    }


    const text =
      String(label);


    return text
      .replace(/[_-]/g, " ")
      .replace(
        /\b\w/g,
        (char) =>
          char.toUpperCase()
      );

  }


  function capitalize(value) {

    if (!value) {
      return "";
    }


    return (
      value.charAt(0).toUpperCase() +
      value.slice(1)
    );

  }


})();