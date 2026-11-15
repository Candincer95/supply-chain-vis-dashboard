import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

CATEGORIES = ['Shipping Mode', 'Order Region', 'Customer Segment']
FEATURES = CATEGORIES + ['Order Item Quantity']
TARGET = 'Late_delivery_risk'
LABELS = {0: 'Not late', 1: 'Late'}


def make_model(kind, depth=2):
    encoder = ColumnTransformer([
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=kind != 'KNN'), CATEGORIES),
        ('quantity', StandardScaler() if kind == 'KNN' else 'passthrough', [FEATURES[-1]])
    ])
    estimator = KNeighborsClassifier(n_neighbors=5, algorithm='brute', n_jobs=1) if kind == 'KNN' else DecisionTreeClassifier(max_depth=depth, min_samples_leaf=100, random_state=42)
    return make_pipeline(encoder, estimator)


def predict(model, training, records, kind):
    if kind == 'Decision tree':
        return {'prediction': model.predict(records[FEATURES])}
    distances, indices = model[-1].kneighbors(model[0].transform(records[FEATURES]))
    votes = training[TARGET].to_numpy()[indices].sum(axis=1)
    return dict(prediction=(votes >= 3).astype(int), distances=distances, indices=indices)


@st.cache_resource(show_spinner=False)
def train_models(df):
    data = df[df['Delivery Status'] != 'Shipping canceled'].copy()
    data['order_date'] = pd.to_datetime(data['order_date'])
    dates = sorted(data['order_date'].unique())
    if len(dates) < 5:
        raise ValueError('At least five distinct order timestamps are needed.')
    first, last = dates[int(.6 * len(dates))], dates[int(.8 * len(dates))]
    initial = data[data['order_date'] < first]
    valid = data[data['order_date'].between(first, last, inclusive='left')]
    fitting = data[data['order_date'] < last].reset_index(drop=True)
    test = data[data['order_date'] >= last].reset_index(drop=True)
    if any(part[TARGET].nunique() != 2 for part in [initial, valid, test]):
        raise ValueError('Each time split must contain both target classes.')
    results, depth_scores = {}, {}
    with threadpool_limits(limits=1):
        for depth in [1, 2]:
            model = make_model('Decision tree', depth).fit(initial[FEATURES], initial[TARGET])
            depth_scores[depth] = balanced_accuracy_score(valid[TARGET], model.predict(valid[FEATURES]))
        depth = max(depth_scores, key=depth_scores.get)
        for kind in ['Decision tree', 'KNN']:
            model = make_model(kind, depth)
            validation_score = depth_scores[depth]
            if kind == 'KNN':
                model.fit(initial[FEATURES], initial[TARGET])
                validation = predict(model, initial, valid, kind)['prediction']
                validation_score = balanced_accuracy_score(valid[TARGET], validation)
            model.fit(fitting[FEATURES], fitting[TARGET])
            result = predict(model, fitting, test, kind)
            result.update(model=model, validation=validation_score)
            results[kind] = result
    majority = int(fitting[TARGET].mode()[0])
    return fitting, test, results, dict(depth=depth, scores=depth_scores, majority=majority,
        baseline=accuracy_score(test[TARGET], [majority] * len(test)), excluded=len(df) - len(data),
        sizes=(len(initial), len(valid), len(test)))


def tree_chart(model, record):
    tree, positions, leaves = model[-1].tree_, {}, []
    questions = [f'{column}<br>{value}?' for column, values in zip(CATEGORIES, model[0].named_transformers_['category'].categories_) for value in values]
    path = set(model[-1].decision_path(model[0].transform(record[FEATURES])).indices)
    def place(node, depth=0):
        if tree.children_left[node] < 0:
            x = len(leaves)
            leaves.append(node)
        else:
            x = (place(tree.children_left[node], depth + 1) + place(tree.children_right[node], depth + 1)) / 2
        positions[node] = (x, -depth)
        return x
    place(0)
    fig, total = go.Figure(), int(tree.n_node_samples[0])
    for node in range(tree.node_count):
        feature, count = tree.feature[node], int(tree.n_node_samples[node])
        values = tree.value[node][0]
        rate = float(values[1] / values.sum())
        label = 'Predict: ' + LABELS[int(values.argmax())]
        if feature >= 0:
            category = feature < len(questions)
            label = questions[feature] if category else f'Quantity <= {tree.threshold[node]:g}?'
            for child, answer in zip([tree.children_left[node], tree.children_right[node]], ['No', 'Yes'] if category else ['Yes', 'No']):
                x, y = positions[node]
                dx, dy = positions[child][0] - x, positions[child][1] - y
                t = np.linspace(0, 1, 25)
                size = int(tree.n_node_samples[child])
                color = '#7c3aed' if child in path else '#cbd5e1'
                fig.add_scatter(x=x + dx * (3*t*t - 2*t**3), y=y + dy*t, mode='lines',
                    line=dict(color=color, width=1 + 22*size/total),
                    hovertemplate=f'{answer}<br>Training records: {size:,}<br>Share: {size/total:.1%}<extra></extra>')
                fig.add_annotation(x=x + dx/2, y=y + dy/2, text=answer, showarrow=False, bgcolor='white', font_color='#475569')
        fig.add_scatter(x=[positions[node][0]], y=[positions[node][1]], mode='markers+text',
            text=[label], textposition='bottom center' if node in leaves else 'top center',
            textfont=dict(size=14, color='#0f172a'),
            marker=dict(size=44, color=[rate], coloraxis='coloraxis', line=dict(color='#7c3aed' if node in path else '#94a3b8', width=4 if node in path else 1)),
            hovertemplate=f'{label}<br>Training records: {count:,}<br>Late rate: {rate:.1%}<br>Share: {count/total:.1%}<br>Selected path: {node in path}<extra></extra>')
    fig.update_layout(template='plotly_white', height=450, showlegend=False,
        margin=dict(l=25, r=85, t=55, b=40), hoverlabel=dict(bgcolor='white', font_color='#0f172a'),
        coloraxis=dict(colorscale='YlOrRd', cmin=0, cmax=1, colorbar=dict(title='Late rate', tickformat='.0%', thickness=14)),
        xaxis=dict(visible=False, range=[-.65, len(leaves)-.35], fixedrange=True),
        yaxis=dict(visible=False, range=[min(y for x,y in positions.values())-.5, .55], fixedrange=True))
    return fig


def neighbor_chart(fitting, result, selected, record):
    neighbors = fitting.iloc[result['indices'][selected]][FEATURES + [TARGET, 'order_date']].copy()
    neighbors['Distance'] = result['distances'][selected]
    neighbors['Outcome'] = neighbors.pop(TARGET).map(LABELS)
    neighbors['Date'] = neighbors.pop('order_date').dt.strftime('%Y-%m-%d')
    neighbors.index = pd.Index(range(1, 6), name='Neighbor')
    angles = np.pi/2 - np.arange(5)*2*np.pi/5
    xs, ys = 1.25*np.cos(angles), 1.25*np.sin(angles)
    colors = ['#ea580c' if value == 'Late' else '#2563eb' for value in neighbors['Outcome']]
    changes = [', '.join(c for c in FEATURES if row[c] != record[c]) or 'None: all inputs match' for _, row in neighbors.iterrows()]
    fig = go.Figure()
    for x, y, color, distance in zip(xs, ys, colors, neighbors['Distance']):
        fig.add_scatter(x=[0,x], y=[0,y], mode='lines', line=dict(color=color, width=2), hoverinfo='skip')
        fig.add_annotation(x=.56*x, y=.56*y, text=f'd = {distance:.3f}', showarrow=False,
                           bgcolor='white', borderpad=3, font=dict(size=12, color='#334155'))
    fig.add_scatter(x=xs, y=ys, mode='markers+text', cliponaxis=False,
        text=[f'Neighbor {i}<br>{value}' for i, value in neighbors['Outcome'].items()],
        textposition='middle center', textfont=dict(color='white', size=12),
        marker=dict(size=86, color=colors, line=dict(color='white', width=2)),
        customdata=neighbors[FEATURES + ['Distance', 'Date']].assign(Differences=changes).to_numpy(),
        hovertemplate='Shipping: %{customdata[0]}<br>Region: %{customdata[1]}<br>Segment: %{customdata[2]}<br>Quantity: %{customdata[3]}<br>Distance: %{customdata[4]:.3f}<br>Date: %{customdata[5]}<br>Different inputs: %{customdata[6]}<extra></extra>')
    fig.add_scatter(x=[0], y=[0], mode='markers+text', text=['Selected<br>test record'],
        textposition='middle center', textfont=dict(color='white', size=13),
        marker=dict(size=105, color='#7c3aed', line=dict(color='white', width=3)),
        hovertemplate='<br>'.join(f'{c}: {record[c]}' for c in FEATURES) + '<extra></extra>')
    votes = int((neighbors['Outcome'] == 'Late').sum())
    outcome = LABELS[int(result['prediction'][selected])]
    fig.add_annotation(x=.5, y=-.10, xref='paper', yref='paper', showarrow=False,
        text=f'<b>{votes} late · {5-votes} not late → Prediction: {outcome}</b>', font=dict(size=14, color='#0f172a'))
    fig.update_layout(template='plotly_white', height=530, showlegend=False, margin=dict(l=20,r=20,t=30,b=70),
        paper_bgcolor='white', plot_bgcolor='white', hoverlabel=dict(bgcolor='white', font_color='#0f172a'),
        xaxis=dict(visible=False, range=[-1.65,1.65], fixedrange=True, constrain='domain'),
        yaxis=dict(visible=False, range=[-1.65,1.65], fixedrange=True, scaleanchor='x', scaleratio=1))
    return fig, neighbors


def render_ml(df):
    st.header('Machine learning')
    st.caption('Four inputs: shipping mode, region, customer segment and quantity. Sidebar filters do not apply.')
    try:
        with st.spinner('Preparing both models…'):
            fitting, test, models, info = train_models(df)
    except ValueError as error:
        st.warning(str(error))
        return
    fingerprint = int(pd.util.hash_pandas_object(test[FEATURES + [TARGET, 'order_date']], index=True).sum())
    if st.session_state.get('delay_sample_fingerprint') != fingerprint:
        st.session_state['delay_sample_rows'] = test.sample(n=min(20,len(test))).index.tolist()
        st.session_state['delay_sample_fingerprint'] = fingerprint
    examples = st.session_state['delay_sample_rows']
    st.caption(f"Test: {len(test):,} later records. Excluded cancellations: {info['excluded']:,}. Menus share {len(examples)} random records, fixed for this session.")
    for number, (kind, result) in enumerate(models.items(), 1):
        st.subheader(f'{number}. {kind} — will this order item be late?')
        prediction = result['prediction']
        values = [accuracy_score(test[TARGET], prediction), info['baseline'], recall_score(test[TARGET], prediction)]
        titles = ['Correct predictions', 'Always predict ' + LABELS[info['majority']].lower(), 'Actual delays found']
        for column, title, value in zip(st.columns(3), titles, values):
            column.metric(title, f'{value:.1%}')
        key = 'delay_example_number' if kind == 'Decision tree' else 'knn_example_number'
        choice = st.selectbox('Record', range(1,len(examples)+1), format_func=lambda n: f'Test record {n}', key=key)
        selected = examples[choice-1]
        st.dataframe(test.loc[[selected],FEATURES], hide_index=True, width='stretch')
        if kind == 'Decision tree':
            fig = tree_chart(result['model'], test.loc[[selected]])
            st.caption('Thicker branches: more fitting records. Warmer nodes: more historical delays. Purple: selected path.')
        else:
            fig, neighbors = neighbor_chart(fitting, result, selected, test.loc[selected,FEATURES])
            st.caption('Center: selected record. Orange: historically late; blue: not late. Link labels show actual distance; circle positions are for readability. Hover to compare inputs.')
        st.plotly_chart(fig, width='stretch', key=f'chart_{number}', config=dict(displayModeBar=False))
        st.write('**Prediction: ' + LABELS[int(prediction[selected])] + '**')
        if st.checkbox('Reveal the actual outcome', key=f'reveal_{number}'):
            actual = int(test.loc[selected,TARGET])
            st.info(f"Actual: {LABELS[actual]}. " + ('Correct prediction.' if actual == prediction[selected] else 'Incorrect prediction.'))
        with st.expander('How it works and evaluation'):
            if kind == 'KNN':
                st.dataframe(neighbors.round(3), width='stretch')
                st.caption('K=5, equal votes. One-hot categories and standardized quantity define Euclidean distance. Zero means matching inputs, not the same order. Distance ties can affect neighbor selection; vote share is not calibrated probability.')
            else:
                st.dataframe(pd.DataFrame(info['scores'].items(), columns=['Depth','Validation balanced accuracy']).round(3), hide_index=True)
                st.caption(f"Selected depth: {info['depth']}. Node counts and colors describe the final fitting set, not test outcomes.")
            st.caption(f"Initial training / validation / test: {info['sizes']}. Refit on {len(fitting):,} earlier records. Validation balanced accuracy: {result['validation']:.1%}.")
            st.dataframe(pd.DataFrame(confusion_matrix(test[TARGET], prediction, labels=[0,1]), index=['Actual: not late','Actual: late'], columns=['Predicted: not late','Predicted: late']))
            st.caption(f"Of late warnings, {precision_score(test[TARGET],prediction,zero_division=0):.1%} were correct. Actual duration and delivery outcomes are not inputs. Cancellations are excluded; missing order IDs limit independence checks. Exploratory results, not deployment validation.")
        st.divider()
