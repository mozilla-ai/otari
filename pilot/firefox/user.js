// Smart Window against the local pilot MLPA (pilot/run.sh up).
// Copy into a fresh profile directory, then start Firefox with that profile
// and sign in to a Mozilla account (Smart Window sends its FxA token to MLPA).
user_pref("browser.smartwindow.enabled", true);
user_pref("browser.smartwindow.tos.consentTime", 1759600000);
user_pref("browser.smartwindow.endpoint", "http://127.0.0.1:8080/v1");
// Needs the Firefox branch smartwindow-mlpa-pilot-endpoint; without it an
// overridden endpoint turns the answers (citations) path off.
user_pref("browser.smartwindow.endpoint.isMLPA", true);
user_pref("browser.smartwindow.searchQuery.endpointURL", "http://127.0.0.1:8080/v1/search");
user_pref("browser.smartwindow.searchTheWebAnswers", true);
user_pref("browser.smartwindow.log", "Debug");
