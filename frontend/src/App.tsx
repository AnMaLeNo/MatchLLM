import { useState } from 'react';
import { Send, Activity, AlertCircle, Cpu, Leaf, Flag, Sliders, LayoutGrid, Award, List, Zap, Eye, EyeOff } from 'lucide-react';

// --- Helpers : conversion score → étiquette lisible ---

type Rating = { label: string; color: string; bgColor: string };

type ConfidenceLevel = 'faible' | 'moyenne' | 'forte';

function getConfidenceRating(level?: ConfidenceLevel): Rating {
  if (level === 'forte') return { label: 'Confiance forte', color: '#22c55e', bgColor: 'rgba(34,197,94,0.15)' };
  if (level === 'moyenne') return { label: 'Confiance moyenne', color: '#f59e0b', bgColor: 'rgba(245,158,11,0.15)' };
  return { label: 'Confiance faible', color: '#ef4444', bgColor: 'rgba(239,68,68,0.15)' };
}

/** TOPSIS : score entre 0 et 1 */
function getTopsisRating(score: number): Rating {
  if (score >= 0.55) return { label: 'Bon', color: '#22c55e', bgColor: 'rgba(34,197,94,0.15)' };
  if (score >= 0.35) return { label: 'Neutre', color: '#9ca3af', bgColor: 'rgba(156,163,175,0.15)' };
  return { label: 'Mauvais', color: '#ef4444', bgColor: 'rgba(239,68,68,0.15)' };
}

/** Analyse sémantique : score de feedback pondéré, environ -5 à 4.2 avec les poids actuels */
function getSemanticRating(score: number): Rating {
  if (score >= 0.4) return { label: 'Bon', color: '#22c55e', bgColor: 'rgba(34,197,94,0.15)' };
  if (score >= -0.1) return { label: 'Neutre', color: '#9ca3af', bgColor: 'rgba(156,163,175,0.15)' };
  return { label: 'Mauvais', color: '#ef4444', bgColor: 'rgba(239,68,68,0.15)' };
}

function formatPercent(value?: number): string {
  if (value === undefined || value === null) return '—';
  return `${Math.round(value * 100)} %`;
}

function formatMetric(value?: number | null, digits = 3): string {
  if (value === undefined || value === null) return '—';
  return value.toFixed(digits);
}

function formatSovereignty(value?: number | null): string {
  if (value === undefined || value === null) return '—';
  return value >= 0.5 ? 'oui' : 'non';
}
import './index.css';

interface AnalyseInfo {
  score_threshold: number;
  nb_reactions_similaires: number;
  nb_modeles_couverts: number;
  max_volume_support: number;
  similarite_moyenne: number;
  support_faible: boolean;
  avertissement?: string | null;
}

interface SemanticMetrics {
  score_semantique: number;
  volume_support: number;
  similarite_moyenne?: number;
  niveau_confiance?: ConfidenceLevel;
}

interface TopsisDetail {
  modele: string;
  score_topsis: number;
  score_semantique?: number | null;
  volume_support?: number | null;
  similarite_moyenne?: number | null;
  niveau_confiance?: ConfidenceLevel;
  kwh_token?: number | null;
  score_souverainete?: number | null;
}

interface TopsisWeights {
  performance_semantique?: number;
  energie?: number;
  souverainete?: number;
}

interface ExcludedModels {
  total: number;
  raisons: Record<string, number>;
  liste?: { modele: string; raison: string; criteres_manquants?: string[] }[];
}

interface ApiResponse {
  message?: string;
  prompt: string;
  recompenses?: Record<string, SemanticMetrics>;
  modele_recommande?: string;
  score_topsis?: number;
  classement_complet?: [string, number][];
  classement_detaille?: TopsisDetail[];
  questions_par_modele?: Record<string, { question: string; score: number }[]>;
  infos_analyse?: AnalyseInfo;
  poids_criteres?: TopsisWeights;
  modeles_exclus?: ExcludedModels;
}

const AHP_PROFILES = {
  precision: [
    [1.0, 5.0, 5.0],
    [0.2, 1.0, 1.0],
    [0.2, 1.0, 1.0]
  ],
  green: [
    [1.0, 0.2, 1.0],
    [5.0, 1.0, 5.0],
    [1.0, 0.2, 1.0]
  ],
  sovereignty: [
    [1.0, 1.0, 0.2],
    [1.0, 1.0, 0.2],
    [5.0, 5.0, 1.0]
  ]
};

function App() {
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // États pour le Routeur TOPSIS
  const [routingMode, setRoutingMode] = useState<'classic' | 'topsis'>('classic');
  const [topsisInputMode, setTopsisInputMode] = useState<'cards' | 'sliders'>('cards');
  const [selectedProfile, setSelectedProfile] = useState<'precision' | 'green' | 'sovereignty'>('precision');
  const [showMatchedQuestions, setShowMatchedQuestions] = useState(true);
  const [showClassicOptions, setShowClassicOptions] = useState(false);
  const [useServerThreshold, setUseServerThreshold] = useState(true);
  const [semanticThreshold, setSemanticThreshold] = useState(0.5);
  // Les sliders vont de 1 à 10
  const [sliderValues, setSliderValues] = useState({ semantic: 5, eco: 5, sovereignty: 5 });

  const calculateAHPMatrix = (sem: number, eco: number, sov: number) => {
    return [
      [1.0, sem / eco, sem / sov],
      [eco / sem, 1.0, eco / sov],
      [sov / sem, sov / eco, 1.0]
    ];
  };

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>, key: keyof typeof sliderValues) => {
    setSliderValues({ ...sliderValues, [key]: parseInt(e.target.value) });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const apiBaseUrl = import.meta.env.DEV ? 'http://localhost:8000' : '';

      let endpoint = '';
      let body: any = { prompt, limit: 1000 };

      if (routingMode === 'classic') {
        endpoint = '/api/evaluer_prompt';
        if (!useServerThreshold) {
          body.score_threshold = semanticThreshold;
        }
      } else {
        endpoint = '/api/meilleur_modele';
        body.matrice_ahp = topsisInputMode === 'cards'
          ? AHP_PROFILES[selectedProfile]
          : calculateAHPMatrix(sliderValues.semantic, sliderValues.eco, sliderValues.sovereignty);
        if (!useServerThreshold) {
          body.score_threshold = semanticThreshold;
        }
      }

      const response = await fetch(`${apiBaseUrl}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        let errorMsg = `Erreur HTTP: ${response.status} ${response.statusText}`;
        try {
          const errData = await response.json();
          if (errData && errData.detail) {
            errorMsg = errData.detail;
          }
        } catch (e) { }
        throw new Error(errorMsg);
      }

      const data = await response.json();
      if (data.message && (!data.recompenses || Object.keys(data.recompenses).length === 0) && !data.modele_recommande) {
        setError(data.message);
        setResult(null);
      } else {
        setResult(data);
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Une erreur est survenue lors de la communication avec l\'API.');
    } finally {
      setLoading(false);
    }
  };


  return (
    <div className="app-container">
      <div className="title-container">
        <h1 className="main-title">MatchLLM</h1>
        <p className="subtitle">Match ton prompt avec l'IA qui lui correspond</p>
      </div>

      <div className="glass-panel">

        {/* Toggle Mode: Classique vs TOPSIS */}
        <div className="mode-tabs">
          <button
            type="button"
            className={`tab-btn ${routingMode === 'classic' ? 'active' : ''}`}
            onClick={() => { setRoutingMode('classic'); setResult(null); }}
          >
            <Activity size={18} /> Analyse Classique
          </button>
          <button
            type="button"
            className={`tab-btn ${routingMode === 'topsis' ? 'active' : ''}`}
            onClick={() => { setRoutingMode('topsis'); setResult(null); }}
          >
            <Zap size={18} /> Analyse par Compromis
          </button>
        </div>

        <form onSubmit={handleSubmit} className="input-group">

          {routingMode === 'topsis' && (
            <div className="topsis-config-panel">
              <div className="sub-mode-tabs">
                <button
                  type="button"
                  className={`sub-tab-btn ${topsisInputMode === 'cards' ? 'active' : ''}`}
                  onClick={() => setTopsisInputMode('cards')}
                >
                  <LayoutGrid size={16} /> Profils Rapides
                </button>
                <button
                  type="button"
                  className={`sub-tab-btn ${topsisInputMode === 'sliders' ? 'active' : ''}`}
                  onClick={() => setTopsisInputMode('sliders')}
                >
                  <Sliders size={16} /> Mode Avancé
                </button>
              </div>

              {topsisInputMode === 'cards' ? (
                <div className="profile-cards">
                  <div
                    className={`profile-card ${selectedProfile === 'precision' ? 'active' : ''}`}
                    onClick={() => setSelectedProfile('precision')}
                  >
                    <Cpu size={24} className="profile-icon icon-blue" />
                    <h3>Performance</h3>
                    <p>Priorité à la qualité, compromis léger sur énergie et souveraineté.</p>
                  </div>
                  <div
                    className={`profile-card green ${selectedProfile === 'green' ? 'active' : ''}`}
                    onClick={() => setSelectedProfile('green')}
                  >
                    <Leaf size={24} className="profile-icon icon-green" />
                    <h3>Énergie</h3>
                    <p>Priorité aux modèles sobres, sans ignorer la qualité.</p>
                  </div>
                  <div
                    className={`profile-card red ${selectedProfile === 'sovereignty' ? 'active' : ''}`}
                    onClick={() => setSelectedProfile('sovereignty')}
                  >
                    <Flag size={24} className="profile-icon icon-red" />
                    <h3>Souveraineté</h3>
                    <p>Priorité aux modèles européens/français, avec garde-fou qualité.</p>
                  </div>
                </div>
              ) : (
                <div className="sliders-container">
                  <div className="slider-group">
                    <div className="slider-header">
                      <span className="slider-label" style={{ color: '#38bdf8' }}><Cpu size={16} /> Compétence Sémantique</span>
                      <span className="slider-value">{sliderValues.semantic}/10</span>
                    </div>
                    <input
                      type="range" min="1" max="10"
                      value={sliderValues.semantic}
                      onChange={(e) => handleSliderChange(e, 'semantic')}
                      className="slider-input blue-slider"
                    />
                  </div>
                  <div className="slider-group">
                    <div className="slider-header">
                      <span className="slider-label" style={{ color: '#10b981' }}><Leaf size={16} /> Économie d'Énergie</span>
                      <span className="slider-value">{sliderValues.eco}/10</span>
                    </div>
                    <input
                      type="range" min="1" max="10"
                      value={sliderValues.eco}
                      onChange={(e) => handleSliderChange(e, 'eco')}
                      className="slider-input green-slider"
                    />
                  </div>
                  <div className="slider-group">
                    <div className="slider-header">
                      <span className="slider-label" style={{ color: '#ef4444' }}><Flag size={16} /> Souveraineté Politique</span>
                      <span className="slider-value">{sliderValues.sovereignty}/10</span>
                    </div>
                    <input
                      type="range" min="1" max="10"
                      value={sliderValues.sovereignty}
                      onChange={(e) => handleSliderChange(e, 'sovereignty')}
                      className="slider-input red-slider"
                    />
                  </div>
                </div>
              )}
            </div>
          )}

          {(routingMode === 'classic' || routingMode === 'topsis') && (
            <div className="classic-options-panel">
              <button
                type="button"
                className={`matched-q-toggle ${showClassicOptions ? 'active' : ''}`}
                onClick={() => setShowClassicOptions(v => !v)}
              >
                <Sliders size={14} /> Options avancées
              </button>

              {showClassicOptions && (
                <div className="classic-options-content">
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={useServerThreshold}
                      onChange={(e) => setUseServerThreshold(e.target.checked)}
                    />
                    Utiliser le seuil par défaut du serveur
                  </label>

                  {!useServerThreshold && (
                    <div className="slider-group">
                      <div className="slider-header">
                        <span className="slider-label" style={{ color: '#38bdf8' }}>
                          <Sliders size={16} /> Seuil de similarité sémantique
                        </span>
                        <span className="slider-value">{semanticThreshold.toFixed(2)}</span>
                      </div>
                      <input
                        type="range"
                        min="0.30"
                        max="0.80"
                        step="0.05"
                        value={semanticThreshold}
                        onChange={(e) => setSemanticThreshold(parseFloat(e.target.value))}
                        className="slider-input blue-slider"
                      />
                      <p className="classic-options-help">
                        Plus le seuil est haut, plus les exemples sont proches, mais moins il y aura de données.
                        Plus il est bas, plus il y aura de données, mais elles seront moins ciblées.
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          <div className="textarea-container">
            <textarea
              placeholder="Que voulez-vous exprimer ? Saisissez le prompt à analyser ici..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              disabled={loading}
              spellCheck="false"
            />
            <button
              type="submit"
              className="submit-btn"
              disabled={loading || !prompt.trim()}
            >
              {loading
                ? (routingMode === 'topsis' ? 'Calcul du compromis...' : 'Analyse en cours...')
                : (routingMode === 'topsis' ? 'Calculer le compromis' : 'Évaluer Sémantiquement')}
              <Send size={16} />
            </button>
          </div>
        </form>

        {error && (
          <div className="error-message">
            <AlertCircle size={20} />
            <span>{error}</span>
          </div>
        )}

        {loading && (
          <div className="loader-container">
            <div className="pulse-bubble"></div>
            <div className="pulse-bubble"></div>
            <div className="pulse-bubble"></div>
            <div className="loader-text" style={{ marginLeft: '20px', color: 'var(--text-secondary)', fontWeight: 600 }}>
              {routingMode === 'topsis' ? 'Calcul du routage TOPSIS...' : 'Analyse sémantique en cours...'}
            </div>
          </div>
        )}

        {/* --- RÉSULTATS : ANALYSE CLASSIQUE --- */}
        {routingMode === 'classic' && result && result.recompenses && Object.keys(result.recompenses).length > 0 && !loading && (
          <div className="results-container">
            <div className="results-header">
              <Activity size={24} color="#38bdf8" />
              <h2>Classement de l'Analyse Sémantique</h2>
              <button
                type="button"
                className={`matched-q-toggle ${showMatchedQuestions ? 'active' : ''}`}
                onClick={() => setShowMatchedQuestions(v => !v)}
                title="Affiche les questions similaires et le détail du calcul"
              >
                {showMatchedQuestions ? <EyeOff size={14} /> : <Eye size={14} />}
                {showMatchedQuestions ? 'Masquer les détails' : 'Voir les détails'}
              </button>
            </div>
            {showMatchedQuestions && (
              <p className="matched-q-hint">
                💡 <strong>Comment lire ces résultats ?</strong><br />
                • <strong>Le score entre parenthèses (ex: 0.850)</strong> indique à quel point ces <strong>questions similaires</strong> se rapprochent de votre prompt <em>(limitées à 3 maximum pour ne pas surcharger l'affichage)</em>.<br />
                • <strong>Le badge de performance</strong> indique le feedback moyen reçu par le modèle sur des demandes proches.<br />
                • <strong>La confiance</strong> dépend du nombre d'exemples similaires disponibles et de leur proximité sémantique.
              </p>
            )}
            {result.infos_analyse && (
              <div className="matched-q-hint" style={{ borderColor: result.infos_analyse.support_faible ? 'rgba(245,158,11,0.35)' : undefined }}>
                <strong>Données utilisées :</strong> {result.infos_analyse.nb_reactions_similaires} réaction{result.infos_analyse.nb_reactions_similaires > 1 ? 's' : ''} similaire{result.infos_analyse.nb_reactions_similaires > 1 ? 's' : ''}, {result.infos_analyse.nb_modeles_couverts} modèle{result.infos_analyse.nb_modeles_couverts > 1 ? 's' : ''}, seuil {result.infos_analyse.score_threshold}.<br />
                Similarité moyenne : {result.infos_analyse.similarite_moyenne.toFixed(3)}.
                {result.infos_analyse.avertissement && (<><br />⚠️ {result.infos_analyse.avertissement}</>)}
              </div>
            )}

            <div className="ranking-section">
              <div className="ranking-list">
                {Object.entries(result.recompenses)
                  // Application de la fonction de tri (décroissant) sur le scalaire score_semantique
                  .sort((a, b) => {
                    const scoreA = a[1].score_semantique || 0;
                    const scoreB = b[1].score_semantique || 0;
                    return scoreB - scoreA;
                  })
                  .map(([modele, metriques], index) => {
                    const score = metriques.score_semantique;
                    const volume = metriques.volume_support;
                    const confidence = getConfidenceRating(metriques.niveau_confiance);

                    return (
                      <div key={modele} className={`ranking-item ${index === 0 ? 'top-1' : ''}`}>
                        <div className="rank">#{index + 1}</div>
                        <div className="model-name" style={{ display: 'flex', flexDirection: 'column' }}>
                          <span>{modele}</span>
                          <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: 'normal', marginTop: '4px' }}>
                            Volume de support : {volume} évaluation{volume > 1 ? 's' : ''}
                            {metriques.similarite_moyenne !== undefined && ` · similarité moyenne ${metriques.similarite_moyenne.toFixed(3)}`}
                          </span>
                          <span style={{
                            alignSelf: 'flex-start', fontSize: '0.75rem', fontWeight: 700, padding: '2px 8px',
                            borderRadius: '20px', color: confidence.color,
                            background: confidence.bgColor, marginTop: '6px'
                          }}>{confidence.label}</span>
                          {showMatchedQuestions && result.questions_par_modele?.[modele] && result.questions_par_modele[modele].length > 0 && (
                            <ul className="inline-matched-questions">
                              {result.questions_par_modele[modele].map((entry, qi) => (
                                <li key={qi} className="inline-matched-question">
                                  <span className="inline-q-index">{qi + 1}</span>
                                  <span className="inline-q-text">
                                    {entry.question}
                                    <span className="inline-q-score">({entry.score.toFixed(3)})</span>
                                  </span>
                                </li>
                              ))}
                            </ul>
                          )}
                        </div>
                        <span style={{
                          fontSize: '0.8rem', fontWeight: 700, padding: '3px 10px',
                          borderRadius: '20px', color: getSemanticRating(score).color,
                          background: getSemanticRating(score).bgColor, whiteSpace: 'nowrap'
                        }}>{getSemanticRating(score).label}</span>
                      </div>
                    );
                  })}
              </div>
            </div>
          </div>
        )}

        {/* --- RÉSULTATS : MODE TOPSIS --- */}
        {routingMode === 'topsis' && result && result.modele_recommande && !loading && (() => {
          const classementTopsis: TopsisDetail[] = result.classement_detaille
            ?? result.classement_complet?.map(([modele, score]) => ({ modele, score_topsis: score }))
            ?? [];
          const winner = classementTopsis[0];
          const winnerRating = getTopsisRating(winner?.score_topsis ?? result.score_topsis ?? 0);

          return (
            <div className="results-container">
              <div className="winner-card compact-winner-card">
                <div className="winner-icon-wrapper">
                  <Award size={40} className="winner-icon" />
                </div>
                <div className="winner-info">
                  <h3 className="winner-title">Meilleur compromis</h3>
                  <div className="winner-model">{result.modele_recommande}</div>
                  <div className="metric-chips">
                    <span className="metric-chip">Compromis {formatMetric(winner?.score_topsis ?? result.score_topsis, 3)}</span>
                    <span className="metric-chip">Sémantique {formatMetric(winner?.score_semantique, 2)}</span>
                    <span className="metric-chip">Énergie {formatMetric(winner?.kwh_token, 3)}</span>
                    <span className="metric-chip">Souveraineté {formatSovereignty(winner?.score_souverainete)}</span>
                    <span style={{
                      fontSize: '0.8rem', fontWeight: 700, padding: '3px 10px',
                      borderRadius: '20px', color: winnerRating.color,
                      background: winnerRating.bgColor, whiteSpace: 'nowrap'
                    }}>{winnerRating.label}</span>
                  </div>
                </div>
              </div>

              {result.poids_criteres && (
                <div className="matched-q-hint compact-info-box">
                  <strong>Poids :</strong> performance {formatPercent(result.poids_criteres.performance_semantique)} · énergie {formatPercent(result.poids_criteres.energie)} · souveraineté {formatPercent(result.poids_criteres.souverainete)}
                </div>
              )}

              {result.infos_analyse && (
                <div className="matched-q-hint compact-info-box" style={{ borderColor: result.infos_analyse.support_faible ? 'rgba(245,158,11,0.35)' : undefined }}>
                  <strong>Données :</strong> {result.infos_analyse.nb_reactions_similaires} réactions similaires · {result.infos_analyse.nb_modeles_couverts} modèles · seuil {result.infos_analyse.score_threshold} · similarité {result.infos_analyse.similarite_moyenne.toFixed(3)}
                  {result.infos_analyse.avertissement && (<><br />⚠️ {result.infos_analyse.avertissement}</>)}
                </div>
              )}

              {result.modeles_exclus && result.modeles_exclus.total > 0 && (
                <div className="matched-q-hint compact-info-box">
                  <strong>Exclusions :</strong> {result.modeles_exclus.total} modèle{result.modeles_exclus.total > 1 ? 's' : ''} sans métriques complètes pour le compromis.
                </div>
              )}

              {classementTopsis.length > 0 && (
                <div className="ranking-section" style={{ marginTop: '25px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '15px', flexWrap: 'wrap' }}>
                    <h3 className="metric-title" style={{ margin: 0 }}><List size={16} /> Classement détaillé</h3>
                    <button
                      type="button"
                      className={`matched-q-toggle ${showMatchedQuestions ? 'active' : ''}`}
                      onClick={() => setShowMatchedQuestions(v => !v)}
                      title="Affiche les questions similaires"
                    >
                      {showMatchedQuestions ? <EyeOff size={14} /> : <Eye size={14} />}
                      {showMatchedQuestions ? 'Masquer les questions' : 'Voir les questions'}
                    </button>
                  </div>
                  {showMatchedQuestions && (
                    <p className="matched-q-hint compact-info-box">
                      Score de compromis = performance sémantique + énergie + souveraineté, pondérées selon le profil choisi.
                    </p>
                  )}
                  <div className="ranking-list">
                    {classementTopsis.map((item, index) => {
                      const confidence = getConfidenceRating(item.niveau_confiance);
                      const rating = getTopsisRating(item.score_topsis);
                      return (
                        <div key={item.modele} className={`ranking-item ${index === 0 ? 'top-1' : ''}`}>
                          <div className="rank">#{index + 1}</div>
                          <div className="model-name" style={{ display: 'flex', flexDirection: 'column' }}>
                            <span>{item.modele}</span>
                            <span className="topsis-detail-line">
                              Sém. {formatMetric(item.score_semantique, 2)} · {item.volume_support ?? '—'} ex. · énergie {formatMetric(item.kwh_token, 3)} · souveraineté {formatSovereignty(item.score_souverainete)}
                            </span>
                            <span style={{
                              alignSelf: 'flex-start', fontSize: '0.75rem', fontWeight: 700, padding: '2px 8px',
                              borderRadius: '20px', color: confidence.color,
                              background: confidence.bgColor, marginTop: '6px'
                            }}>{confidence.label}</span>
                            {showMatchedQuestions && result.questions_par_modele?.[item.modele] && result.questions_par_modele[item.modele].length > 0 && (
                              <ul className="inline-matched-questions">
                                {result.questions_par_modele[item.modele].map((entry: { question: string; score: number }, qi: number) => (
                                  <li key={qi} className="inline-matched-question">
                                    <span className="inline-q-index">{qi + 1}</span>
                                    <span className="inline-q-text">
                                      {entry.question}
                                      <span className="inline-q-score">({entry.score.toFixed(3)})</span>
                                    </span>
                                  </li>
                                ))}
                              </ul>
                            )}
                          </div>
                          <span style={{
                            fontSize: '0.8rem', fontWeight: 700, padding: '3px 10px',
                            borderRadius: '20px', color: rating.color,
                            background: rating.bgColor, whiteSpace: 'nowrap'
                          }}>{formatMetric(item.score_topsis, 3)}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          );
        })()}
      </div>
    </div>
  );
}

export default App;
