(async function () {
  const receipt = document.getElementById('receipt-check');
  const chart = document.getElementById('discrimination-chart');
  const staticFinal = {
    baseline: {average_precision: 0.7028635371374776, roc_auc: 0.5539795346696763},
    model: {average_precision: 0.7658273276523343, roc_auc: 0.6537340524530595}
  };

  function drawDiscrimination(data) {
    const rows = [
      ['ROC AUC', data.baseline.roc_auc, data.model.roc_auc],
      ['Average precision', data.baseline.average_precision, data.model.average_precision]
    ];
    chart.innerHTML = '<h3>Whole-ranking metrics</h3>' + rows.map(([label, baseline, model]) => `
      <div class="chart-group">
        <strong>${label}</strong>
        <div class="bar-row"><span>Baseline</span><div class="track"><div class="fill baseline" style="width:${baseline * 100}%"></div></div><b>${baseline.toFixed(3)}</b></div>
        <div class="bar-row"><span>Model</span><div class="track"><div class="fill model" style="width:${model * 100}%"></div></div><b>${model.toFixed(3)}</b></div>
      </div>`).join('');
  }

  drawDiscrimination(staticFinal);

  try {
    const response = await fetch('../work/outputs/capstone_metrics.json', {cache: 'no-store'});
    if (!response.ok) throw new Error('metric receipt unavailable');
    const data = await response.json();
    if (data.status !== 'verified_warehouse_run' || data.sealed_final.model.n !== 93048) {
      throw new Error('unexpected metric receipt');
    }
    drawDiscrimination(data.sealed_final);
    receipt.textContent = 'Aggregate metric receipt loaded and verified in this page.';
    receipt.classList.add('verified');
  } catch (error) {
    receipt.textContent = 'Static verified results shown; live receipt check was unavailable.';
  }
}());
